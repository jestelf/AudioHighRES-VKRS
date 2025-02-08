import torch
import torch.distributed as dist
from torch.nn.modules import Module
from torch.autograd import Variable

def _flatten_dense_tensors(tensors):
    """
    Преобразует список тензоров в один плоский тензор для эффективной передачи между узлами.
    Все тензоры должны быть одного типа.
    
    Аргументы:
        tensors (Iterable[Tensor]): список плотных тензоров.
    
    Возвращает:
        Один одномерный тензор, содержащий все входные тензоры.
    """
    if len(tensors) == 1:
        return tensors[0].contiguous().view(-1)
    flat = torch.cat([t.contiguous().view(-1) for t in tensors], dim=0)
    return flat

def _unflatten_dense_tensors(flat, tensors):
    """
    Разделяет плоский тензор обратно на исходные тензоры.
    
    Аргументы:
        flat (Tensor): сплющенный тензор.
        tensors (Iterable[Tensor]): список тензоров, размеры которых используются для разбиения.
    
    Возвращает:
        Кортеж тензоров, соответствующих исходным размерам.
    """
    outputs = []
    offset = 0
    for tensor in tensors:
        numel = tensor.numel()
        outputs.append(flat.narrow(0, offset, numel).view_as(tensor))
        offset += numel
    return tuple(outputs)

class DistributedDataParallel(Module):
    """
    Класс для параллельного обучения нейросети на нескольких GPU с помощью PyTorch DistributedDataParallel.
    """

    def __init__(self, module):
        """
        Аргументы:
            module (torch.nn.Module): модель для обучения.
        """
        super(DistributedDataParallel, self).__init__()

        # Определяем, нужно ли предупреждать о медленной работе в режиме half precision с GLOO
        if not hasattr(dist, '_backend'):
            self.warn_on_half = True
        else:
            self.warn_on_half = dist._backend == dist.dist_backend.GLOO

        self.module = module

        # Передаём параметры модели всем процессам
        for p in self.module.state_dict().values():
            if not torch.is_tensor(p):
                continue
            dist.broadcast(p, 0)

        def allreduce_params():
            """
            Функция для выполнения градиентного all-reduce (усреднения градиентов между всеми процессами).
            """
            if self.needs_reduction:
                self.needs_reduction = False
                buckets = {}

                # Разделяем параметры по их типам данных
                for param in self.module.parameters():
                    if param.requires_grad and param.grad is not None:
                        tp = type(param.data)
                        if tp not in buckets:
                            buckets[tp] = []
                        buckets[tp].append(param)

                # Предупреждение о возможной медленной работе с half precision
                if self.warn_on_half and torch.cuda.HalfTensor in buckets:
                    print("WARNING: gloo dist backend for half parameters may be extremely slow." +
                          " It is recommended to use the NCCL backend in this case.")
                    self.warn_on_half = False

                # Применяем all-reduce к градиентам
                for tp in buckets:
                    bucket = buckets[tp]
                    grads = [param.grad.data for param in bucket]
                    coalesced = _flatten_dense_tensors(grads)
                    dist.all_reduce(coalesced)
                    coalesced /= dist.get_world_size()
                    for buf, synced in zip(grads, _unflatten_dense_tensors(coalesced, grads)):
                        buf.copy_(synced)

        # Устанавливаем хук для автоматического усреднения градиентов после backward
        for param in list(self.module.parameters()):
            def allreduce_hook(*unused):
                param._execution_engine.queue_callback(allreduce_params)
            if param.requires_grad:
                param.register_hook(allreduce_hook)

    def forward(self, *inputs, **kwargs):
        """
        Переопределённый forward, устанавливающий флаг необходимости усреднения градиентов.
        """
        self.needs_reduction = True
        return self.module(*inputs, **kwargs)

def apply_gradient_allreduce(module):
    """
    Функция модифицирует переданный модуль, добавляя к нему градиентное усреднение (all-reduce).
    
    Аргументы:
        module (torch.nn.Module): модуль, к которому применяется усреднение градиентов.
    
    Возвращает:
        torch.nn.Module: модуль с добавленной логикой all-reduce.
    """
    if not hasattr(dist, '_backend'):
        module.warn_on_half = True
    else:
        module.warn_on_half = dist._backend == dist.dist_backend.GLOO

    # Передаём параметры модели всем процессам
    for p in module.state_dict().values():
        if not torch.is_tensor(p):
            continue
        dist.broadcast(p, 0)

    def allreduce_params():
        """
        Выполняет усреднение градиентов между процессами после backward.
        """
        if module.needs_reduction:
            module.needs_reduction = False
            buckets = {}

            # Группируем параметры по их типам данных
            for param in module.parameters():
                if param.requires_grad and param.grad is not None:
                    tp = param.data.dtype
                    if tp not in buckets:
                        buckets[tp] = []
                    buckets[tp].append(param)

            # Предупреждение при использовании GLOO с half precision
            if module.warn_on_half and torch.cuda.HalfTensor in buckets:
                print("WARNING: gloo dist backend for half parameters may be extremely slow." +
                      " It is recommended to use the NCCL backend in this case.")
                module.warn_on_half = False

            # Применяем all-reduce к градиентам
            for tp in buckets:
                bucket = buckets[tp]
                grads = [param.grad.data for param in bucket]
                coalesced = _flatten_dense_tensors(grads)
                dist.all_reduce(coalesced)
                coalesced /= dist.get_world_size()
                for buf, synced in zip(grads, _unflatten_dense_tensors(coalesced, grads)):
                    buf.copy_(synced)

    # Устанавливаем хуки для автоматического усреднения градиентов
    for param in list(module.parameters()):
        def allreduce_hook(*unused):
            Variable._execution_engine.queue_callback(allreduce_params)
        if param.requires_grad:
            param.register_hook(allreduce_hook)

    def set_needs_reduction(self, input, output):
        """
        Устанавливает флаг, сигнализирующий о необходимости all-reduce.
        """
        self.needs_reduction = True

    module.register_forward_hook(set_needs_reduction)
    return module
