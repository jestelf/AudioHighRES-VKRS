import time
import torch
import sys
import subprocess

# Получаем аргументы командной строки, исключая имя скрипта
argslist = list(sys.argv)[1:]

# Определяем количество доступных GPU
num_gpus = torch.cuda.device_count()
argslist.append('--n_gpus={}'.format(num_gpus))  # Передаём количество GPU в аргументы

# Список процессов-воркеров
workers = []

# Уникальный идентификатор для группы процессов (основан на текущем времени)
job_id = time.strftime("%Y_%m_%d-%H%M%S")
argslist.append("--group_name=group_{}".format(job_id))  # Название группы процессов

# Запускаем процесс на каждом доступном GPU
for i in range(num_gpus):
    argslist.append('--rank={}'.format(i))  # Добавляем аргумент ранга процесса

    # Логирование вывода в файл для всех процессов, кроме первого (stdout для него остаётся в терминале)
    stdout = None if i == 0 else open("logs/{}_GPU_{}.log".format(job_id, i), "w")

    print(argslist)  # Выводим аргументы запуска (для отладки)

    # Запускаем процесс с переданными аргументами
    p = subprocess.Popen([str(sys.executable)] + argslist, stdout=stdout)

    # Добавляем процесс в список воркеров
    workers.append(p)

    # Удаляем последний аргумент (--rank), чтобы при следующей итерации он добавился с новым значением
    argslist = argslist[:-1]

# Ожидаем завершения всех процессов
for p in workers:
    p.wait()
