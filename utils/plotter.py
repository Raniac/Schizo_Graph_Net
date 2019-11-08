def plot_acc_loss(log_path):
    import matplotlib.pyplot as plt

    with open(log_path, 'r') as f:
        lines = f.readlines()
    
    acc_list = []
    loss_list = []
    for line in lines:
        if 'Epoch' not in line:
            continue
        loss_loc = line.find('Loss')
        loss = float(line[loss_loc+6:loss_loc+12])
        loss_list.append(loss)
        acc_loc = line.find('Test')
        acc = float(line[acc_loc+6:acc_loc+12])
        acc_list.append(acc)
    
    iters = range(1, len(acc_list)+1)
    plt.figure()
    plt.plot(iters, acc_list, 'r', label='test_acc')
    plt.plot(iters, loss_list, 'b', label='train_loss')
    plt.grid(True)
    plt.xlabel('Epochs')
    plt.ylabel('acc/loss')
    plt.legend(loc='upper right')
    plt.savefig(log_path[:-3] + 'png', dpi=300)

if __name__ == "__main__":
    log_path = 'logs/train_val_2019-11-06-08-18-16.log'
    plot_acc_loss(log_path)