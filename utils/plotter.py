def plot_acc_loss(log_path):
    import matplotlib.pyplot as plt

    with open(log_path, 'r') as f:
        lines = f.readlines()
    
    train_acc_list = []
    train_loss_list = []
    test_acc_list = []
    test_loss_list = []
    for line in lines:
        if 'Epoch' not in line:
            continue
        train_loss_loc = line.find('Train Loss')
        train_loss = float(line[train_loss_loc+12:train_loss_loc+18])
        train_loss_list.append(train_loss)
        train_acc_loc = line.find('Train Accuracy')
        train_acc = float(line[train_acc_loc+16:train_acc_loc+22])
        train_acc_list.append(train_acc)
        test_loss_loc = line.find('Test Loss')
        test_loss = float(line[test_loss_loc+11:test_loss_loc+17])
        test_loss_list.append(test_loss)
        test_acc_loc = line.find('Test Accuracy')
        test_acc = float(line[test_acc_loc+15:test_acc_loc+21])
        test_acc_list.append(test_acc)
    
    iters = range(1, len(train_acc_list)+1)
    plt.figure()
    plt.plot(iters, train_acc_list, 'r', label='Train Accuracy')
    plt.plot(iters, train_loss_list, 'r', label='Train Loss')
    plt.plot(iters, test_acc_list, 'b', label='Test Accuracy')
    plt.plot(iters, test_loss_list, 'b', label='Test Loss')
    plt.grid(True)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy/Loss')
    plt.legend(loc='upper right')
    plt.savefig(log_path[:-3] + 'png', dpi=300)

if __name__ == "__main__":
    log_path = 'logs/train_val.log'
    plot_acc_loss(log_path)