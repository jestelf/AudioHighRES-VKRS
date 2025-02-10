#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для анализа логов обучения.
Читает CSV-файл с метриками (например, training_metrics.csv) из резервной папки,
строит графики (Average Loss, Delta Loss, Learning Rate vs. Epoch),
и генерирует отчёт с анализом динамики обучения.
Графики и отчёт сохраняются в папке "plots" внутри резервной папки.
"""

import os
import csv
import matplotlib.pyplot as plt

def read_metrics(csv_filename):
    epochs, avg_losses, deltas, lrs = [], [], [], []
    with open(csv_filename, "r", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            epochs.append(int(row["epoch"]))
            avg_losses.append(float(row["avg_loss"]))
            deltas.append(float(row["delta"]))
            lrs.append(float(row["learning_rate"]))
    return epochs, avg_losses, deltas, lrs

def plot_metrics(epochs, avg_losses, deltas, lrs, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # График Average Loss vs. Epoch
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, avg_losses, marker="o", label="Average Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Average Loss")
    plt.title("Average Loss vs. Epoch")
    plt.legend()
    plt.grid(True)
    loss_path = os.path.join(output_dir, "loss_vs_epoch.png")
    plt.savefig(loss_path)
    plt.close()
    
    # График Delta Loss vs. Epoch
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, deltas, marker="o", color="orange", label="Delta Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Delta Loss")
    plt.title("Delta Loss vs. Epoch")
    plt.legend()
    plt.grid(True)
    delta_path = os.path.join(output_dir, "delta_loss_vs_epoch.png")
    plt.savefig(delta_path)
    plt.close()
    
    # График Learning Rate vs. Epoch
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, lrs, marker="o", color="green", label="Learning Rate")
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate vs. Epoch")
    plt.legend()
    plt.grid(True)
    lr_path = os.path.join(output_dir, "lr_vs_epoch.png")
    plt.savefig(lr_path)
    plt.close()
    
    print(f"Графики сохранены в папке: {output_dir}")

def analyze_metrics(epochs, avg_losses, deltas):
    report_lines = []
    best_epoch = epochs[avg_losses.index(min(avg_losses))]
    report_lines.append(f"Лучшая эпоха: {best_epoch} с Average Loss = {min(avg_losses):.6f}")
    report_lines.append(f"Финальная эпоха: {epochs[-1]} с Average Loss = {avg_losses[-1]:.6f}")
    
    avg_delta = sum(abs(d) for d in deltas) / len(deltas)
    report_lines.append(f"Средняя абсолютная дельта loss: {avg_delta:.6f}")
    if avg_delta < 1e-4:
        report_lines.append("Обучение стабилизировано. Возможно, достигнут предел улучшения.")
    else:
        report_lines.append("Наблюдаются значимые изменения в loss между эпохами. Рассмотрите возможность корректировки learning rate или увеличения latent_dim для дальнейших улучшений.")
    return "\n".join(report_lines)

def main():
    csv_filename = input("Введите полный путь к CSV-файлу с метриками: ").strip()
    if not os.path.exists(csv_filename):
        print("Файл не найден!")
        return
    backup_dir = os.path.dirname(csv_filename)
    plots_dir = os.path.join(backup_dir, "plots")
    
    epochs, avg_losses, deltas, lrs = read_metrics(csv_filename)
    plot_metrics(epochs, avg_losses, deltas, lrs, plots_dir)
    report = analyze_metrics(epochs, avg_losses, deltas)
    report_path = os.path.join(plots_dir, "report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print("Отчёт сгенерирован:")
    print(report)

if __name__ == "__main__":
    main()
