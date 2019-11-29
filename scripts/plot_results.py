import numpy as np
import matplotlib.pyplot as plt
import sys




def plot_results(plot_data_name, title):
  [train_losses, test_losses, train_accs, test_accs] = np.load("plot_data/{}.npy".format(plot_data_name))
  epoch_range = range(1, 45 + 1)
  train_losses = train_losses[:45]
  test_losses = test_losses[:45]
  train_accs = train_accs[:45]
  test_accs = test_accs[:45]
  max_test_acc_epoch = np.argmax(test_accs)
  max_test_acc = test_accs[max_test_acc_epoch]
  max_test_acc_point = (max_test_acc_epoch + 1, round(max_test_acc, 3))
  max_test_acc_text_point = (max_test_acc_point[0] - 5, max_test_acc_point[1] + 0.003)

  plt.plot(epoch_range, train_losses, label="Train")
  plt.plot(epoch_range, test_losses, label="Test")
  plt.xlabel("Epoch")
  plt.ylabel("Loss")
  plt.title("{} Loss".format(title))
  plt.legend()
  plt.show()

  plt.plot(epoch_range, train_accs, label="Train")
  plt.plot(epoch_range, test_accs, label="Test")
  plt.plot((max_test_acc_epoch + 1), max_test_acc, "g*")
  plt.annotate(str(max_test_acc_point), xy=max_test_acc_point, xytext=max_test_acc_text_point)
  plt.xlabel("Epoch")
  plt.ylabel("Accuracy")
  plt.title("{} Accuracy".format(title))
  plt.legend()
  plt.show()



def plot_lr():
  test_loss_thousanth = np.load("results/from-eddie/20-0.001.npy")[1]
  test_loss_hundredth = np.load("results/from-eddie/20-0.01.npy")[1]
  test_loss_tenth = np.load("results/20-0.1.npy")[1]
  test_loss_one = np.load("results/20-1.0.npy")[1]

  epoch_range = range(1, 501)

  plt.plot(epoch_range, test_loss_thousanth, label="0.001")
  plt.plot(epoch_range, test_loss_hundredth, label="0.01")
  plt.plot(epoch_range, test_loss_tenth, label="0.1")
  plt.plot(epoch_range, test_loss_one, label="1")

  plt.xlabel("Epoch")
  plt.ylabel("Test Loss (RMSE)")
  #plt.title("Performance of 20-Factor Model with Different Learning Rates")
  plt.legend(title="Learning Rate")
  plt.show()


def plot_num_features():
  test_loss_ten = np.load("results/10-0.1.npy")[1]
  test_loss_twenty = np.load("results/20-0.1.npy")[1]

  epoch_range = range(1, 501)

  plt.plot(epoch_range, test_loss_ten, label="10")
  plt.plot(epoch_range, test_loss_twenty, label="20")

  plt.xlabel("Epoch")
  plt.ylabel("Test Loss (RMSE)")
  #plt.title("10-Feature Model vs. 20-Feature Model")
  plt.legend(title="Embedding Size")
  plt.show()


def get_results(file):
  return np.load("results/{}.npy".format(file))[1]

def get_loss_at_epoch(file, epoch):
  return np.load("results/{}.npy".format(file))[1][epoch - 1]

def get_perf(file):
  return min(np.load("results/{}.npy".format(file))[1])


def get_epoch(file):
  return np.argmin(np.load("results/{}.npy".format(file))[1])


