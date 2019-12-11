import sys
import numpy as np
import torch
import pandas as pd
import pickle
import math
import time



EPOCHS = 500
BATCH_SIZE = 32
EVAL_BATCH_SIZE = 500
is_cuda = True
is_biased = True
verbose = False


# Load and preprocess data
train = pd.read_pickle("../data/ml-1m-split/train.pkl")
val = pd.read_pickle("../data/ml-1m-split/val.pkl")

train = train.sample(frac=1) # Shuffle the training set


# Data constants
NUM_USERS = 6040
NUM_ITEMS = 3706

GLOBAL_AVERAGE = train["rating"].mean()

cos = torch.nn.CosineSimilarity()


class MatrixFactorization(torch.nn.Module):

  def __init__(self, num_users, num_items, num_factors):
    super().__init__()
    self.user_factors = torch.nn.Embedding(num_users, num_factors, max_norm=1, sparse=True)
    self.item_factors = torch.nn.Embedding(num_items, num_factors, max_norm=1, sparse=True)

    self.user_factors.weight.data.uniform_(-0.25, 0.25)
    self.item_factors.weight.data.uniform_(-0.25, 0.25)

  def forward(self, users, items):
    return (cos(self.user_factors(users), self.item_factors(items)) * 2.25) + 2.75



class BiasedMatrixFactorization(torch.nn.Module):

  def __init__(self, num_users, num_items, num_factors):
    super().__init__()
    self.user_factors = torch.nn.Embedding(num_users, num_factors, sparse=False)
    self.item_factors = torch.nn.Embedding(num_items, num_factors, sparse=False)
    self.user_biases = torch.nn.Embedding(num_users, 1, sparse=False)
    self.item_biases = torch.nn.Embedding(num_items, 1, sparse=False)

    self.user_factors.weight.data.uniform_(-0.25, 0.25)
    self.item_factors.weight.data.uniform_(-0.25, 0.25)
    self.user_biases.weight.data.uniform_(-0.25, 0.25)
    self.item_biases.weight.data.uniform_(-0.25, 0.25)

  def forward(self, users, items):
    return GLOBAL_AVERAGE + self.user_biases(users).squeeze(dim=1) + self.item_biases(items).squeeze(dim=1) \
       + torch.diagonal(torch.mm(self.user_factors(users), torch.transpose(self.item_factors(items), 0, 1)))



def train_model(num_factors, learning_rate, weight_decay):

  if is_cuda:
    if is_biased:
      model = BiasedMatrixFactorization(NUM_USERS, NUM_ITEMS, num_factors).cuda()
    else:
      model = MatrixFactorization(NUM_USERS, NUM_ITEMS, num_factors).cuda()
    long_type = torch.cuda.LongTensor
    float_type = torch.cuda.FloatTensor
    torch.backends.cudnn.benchmark=True
  else:
    if is_biased:
      model = BiasedMatrixFactorization(NUM_USERS, NUM_ITEMS, num_factors)
    else:
      model = MatrixFactorization(NUM_USERS, NUM_ITEMS, num_factors)
    long_type = torch.LongTensor
    float_type = torch.FloatTensor

  loss_fn = torch.nn.MSELoss()
  optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

  def evaluate_model(df):
    EVAL_BATCH_SIZE = 500
    losses = []
    for j in range(0, math.ceil(df.shape[0] / EVAL_BATCH_SIZE)):
      batch = df.iloc[j * EVAL_BATCH_SIZE : (j + 1) * EVAL_BATCH_SIZE]
      users = torch.Tensor(batch["user"].values).type(long_type)
      items = torch.Tensor(batch["item"].values).type(long_type)
      ratings = torch.Tensor(batch["rating"].values).type(float_type)
      predictions = model(users, items)
      loss = loss_fn(predictions, ratings)
      losses.append(loss.item())

    return np.sqrt(np.mean(losses))


  train_losses = np.zeros(EPOCHS)
  val_losses = np.zeros(EPOCHS)

  for i in range(EPOCHS):
    n = 0
    # time1 = time.time()
    # time2 = time1
    for j in range(0, math.ceil(train.shape[0] / BATCH_SIZE)):
    #for row in train.itertuples(index=False):
      # time1 = time.time()
      # print("Time Elapsed 1: {}".format(time1 - time2))
      batch = train.iloc[j * BATCH_SIZE : (j + 1) * BATCH_SIZE]
      users = torch.Tensor(batch["user"].values).type(long_type)
      items = torch.Tensor(batch["item"].values).type(long_type)
      ratings = torch.Tensor(batch["rating"].values).type(float_type)
      predictions = model(users, items)
      loss = loss_fn(predictions, ratings)
      loss.backward()
      optimizer.step()
      model.zero_grad()
      
      if verbose and (n % 1000 == 0):
        print(n)
        print("Loss: {}".format(loss.item()))
        # print("User embedding: {}".format(model.user_factors(users)))
        # print("Movie embedding: {}".format(model.item_factors(items)))
        # if is_biased:
        #   print("User bias: {}".format(model.user_biases(users)))
        #   print("Movie bias: {}".format(model.item_biases(items)))
        # print("Prediction: {}".format(predictions[0]))
        # print("Rating: {}".format(ratings[0]))
        # print("Diff: {}\n".format(abs(predictions[0] - ratings[0])))
      n += 1

      # time2 = time.time()
      # print("Time Elapsed 2: {}".format(time2 - time1))
      
    train_loss = evaluate_model(train)
    val_loss = evaluate_model(val)

    # Calling .item() releases the copy of the computational graph stored in loss
    train_losses[i] = train_loss.item()
    val_losses[i] = val_loss.item()

    print("============== EPOCH {} ==============\nTrain RMSE = {}\nValidation RMSE = {}\n"\
      .format(i + 1, train_loss, val_loss))
    
    result_path = "../results/biasedmf_{}_{}_{}".format(num_factors, learning_rate, weight_decay)
    #model_path = "../models/{}_{}_{}.pt".format(num_factors, learning_rate, is_biased)

    np.save(result_path, [train_losses, val_losses])
    #torch.save(model.state_dict(), model_path)

    





if __name__ == '__main__':
  args = sys.argv
  if len(args) >= 4:
    train_model(int(args[1]), float(args[2]), (float(args[3])))

     





