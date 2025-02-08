# plot_loss_and_summary.py
import re
import matplotlib.pyplot as plt
import numpy as np

def parse_log(log_file_path):
    epoch_numbers = []
    loss_values = []
    pattern = r"Epoch (\d+): Train Loss = ([0-9\.]+), Val Loss = ([0-9\.]+)"
    with open(log_file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = re.search(pattern, line)
            if m:
                epoch_numbers.append(int(m.group(1)))
                loss_values.append(float(m.group(3)))
    return epoch_numbers, loss_values

def plot_loss(epochs, losses, output_image="loss_plot.png"):
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, losses, marker="o", linestyle="-", color="b")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.title("Validation Loss vs Epoch")
    plt.grid(True)
    plt.savefig(output_image)
    plt.show()
    print("Plot saved as", output_image)

if __name__ == '__main__':
    log_file_path = "training.log"
    epochs, losses = parse_log(log_file_path)
    if epochs:
        plot_loss(epochs, losses)
    else:
        print("No loss data found in log.")
