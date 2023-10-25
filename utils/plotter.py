import matplotlib.pyplot as plt
import os

def plot_loss(label, loss, save_path):
    path = os.path.join(save_path, f'{label}.png')
    plt.figure()
    plt.xlabel('Epoch')
    plt.plot(loss, label=label)
    plt.legend()
    plt.savefig(path)
    plt.close()


if __name__ == '__main__':
    all_val_loss = [0.5,0.4,0.35,0.22,0.10,0.09]
    save_path = './'
    plot_loss('test_fig', all_val_loss, save_path)