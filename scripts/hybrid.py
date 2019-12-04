import sys
import numpy as np
import torch
from torch import nn
import pandas as pd
import pickle


EPOCHS = 500
MAX_BATCH_SIZE = 32
ITEM_HIDDIM = 250
is_cuda = False
verbose = True


if is_cuda:
    long_type = torch.cuda.LongTensor
    float_type = torch.cuda.FloatTensor
    torch.backends.cudnn.benchmark=True
else:
  long_type = torch.LongTensor
  float_type = torch.FloatTensor


# Load and preprocess data
print("Loading training and validation data...")
train = pd.read_pickle("../data/ml-20m-split/train.pkl")
train = train.sample(frac=1) # Shuffle the training set

# with open("../data/ml-20m-split/train_batched.pkl", "rb") as f:
#   train_batches = pickle.load(f)
train_batches = np.load("../data/ml-20m-split/train_batched_numpy.npy", allow_pickle=True)
np.random.shuffle(train_batches) # Shuffle the batches

val = pd.read_pickle("../data/ml-20m-split/val.pkl")

NUM_BATCHES = len(train_batches)

print("Loading item inputs...")
all_item_inputs = np.load("../data/item_inputs.npy")
ITEM_INPUT_DIM = len(all_item_inputs[0])

# Data constants
NUM_USERS = train.shape[0]
NUM_ITEMS = len(all_item_inputs)
GLOBAL_AVERAGE = train["rating"].mean()


print("Loading user items...")
all_user_items = np.load("../data/user_items_numpy.npy", allow_pickle=True)


# with open("../data/user_items.pkl", "rb") as f:
#   all_user_items = pickle.load(f)



class MatrixFactorization(nn.Module):
  def __init__(self, num_users, num_items, num_factors):
    super().__init__()
    self.user_factors = nn.Embedding(num_users, num_factors, sparse=True)
    self.item_factors = nn.Embedding(num_items, num_factors, sparse=True)
    
  def forward(self, users, items):
    return torch.diagonal(torch.mm(self.user_factors(users), torch.transpose(self.item_factors(items), 0, 1)))
    


class ContentRecommender(nn.Module):
  def __init__(self, num_factors, item_hiddim, user_hiddim):
    super().__init__()
    self.outdim = num_factors

    # Item model
    self.item_hiddim = item_hiddim # 250
    self.item_model = nn.Sequential(
      nn.Linear(ITEM_INPUT_DIM, self.item_hiddim),
      nn.ReLU(),
      nn.Linear(self.item_hiddim, self.outdim)
    )

    # User model
    self.user_hiddim = user_hiddim # num_factors
    self.user_lstm = nn.LSTM(self.outdim, self.user_hiddim, batch_first=True)


  def forward(self, users, items):
    batch_size = users.shape[0]
    #print(np.asarray(users).astype(int))
    print("seq_len: {}".format(len(all_user_items[users[0]])))
    #print("{}, {}".format(users.shape[0], len(all_user_items[users[0]])))
    # print(len(all_user_items[users[0]]))
    assert(users.shape[0] == items.shape[0])
    user_item_indices = np.concatenate(all_user_items[np.asarray(users)])
    user_item_inputs = torch.Tensor(all_item_inputs[user_item_indices]).type(float_type)
    #print("{}, {}".format(len(user_item_indices), user_item_inputs.shape))
    #user_item_inputs_indices = np.arange(user_item_inputs.shape[0]).reshape(users.shape[0], -1).T.reshape(-1)
    #print(self.item_model(user_item_inputs))
    #print(self.item_model(user_item_inputs).shape)
    user_inputs = self.item_model(user_item_inputs).view(batch_size, -1, self.outdim)
    #user_inputs = self.item_model(user_item_inputs)[user_item_inputs_indices].view(-1, BATCH_SIZE, self.outdim)
    item_inputs = torch.Tensor(all_item_inputs[items])
    
    item_embeddings = self.item_model(item_inputs)
    (lstm_outputs, _) = self.user_lstm(user_inputs)
    
    #print("Outputs shape: {}".format(lstm_outputs.permute(1, 0, 2).shape))
    user_embeddings = lstm_outputs.permute(1, 0, 2)[-1]
    # if item_embeddings.shape != user_embeddings.shape:
    #   print("User embedding: {}".format(user_embeddings.shape))
    #   print("Item embedding: {}".format(item_embeddings.shape))
    assert(item_embeddings.shape == user_embeddings.shape)
    return torch.diagonal(torch.mm(user_embeddings, torch.transpose(item_embeddings, 0, 1)))



class HybridRecommender(nn.Module):
  def __init__(self, num_users, num_items, num_factors, item_hiddim, user_hiddim):
    super().__init__()

    #self.mf_model = MatrixFactorization(num_users, num_items, num_factors)
    self.content_model = ContentRecommender(num_factors, item_hiddim, user_hiddim)

    self.user_biases = nn.Embedding(num_users, 1, sparse=True)
    self.item_biases = nn.Embedding(num_items, 1, sparse=True)


  def forward(self, users, items):
    return GLOBAL_AVERAGE + self.user_biases(users).squeeze(dim=1) + self.item_biases(items).squeeze(dim=1) \
      + self.content_model(users, items)



def train_model(num_factors, learning_rate):
  print("Training model...")
  if is_cuda:
    model = HybridRecommender(NUM_USERS, NUM_ITEMS, num_factors, ITEM_HIDDIM, num_factors).cuda()   
  else:
    model = HybridRecommender(NUM_USERS, NUM_ITEMS, num_factors, ITEM_HIDDIM, num_factors)
    
  loss_fn = nn.MSELoss()
  optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


  def evaluate_model(df):
    users = torch.Tensor(df["user"].values).type(long_type)
    items = torch.Tensor(df["item"].values).type(long_type)
    ratings = torch.Tensor(df["rating"].values).type(float_type)
    predictions = model(users, items)
    return torch.sqrt(loss_fn(predictions, ratings))

  train_losses = np.zeros(EPOCHS)
  val_losses = np.zeros(EPOCHS)

  j = 0
  for i in range(EPOCHS):
    for batch in train_batches:
      j += 1
      if j % 1000 == 0:
        print("Batch {} of {}".format(j, NUM_BATCHES))
      users = torch.Tensor(batch[:, 0]).type(long_type)
      items = torch.Tensor(batch[:, 1]).type(long_type)
      ratings = torch.Tensor(batch[:, 2]).type(float_type)
      predictions = model(users, items)
      loss = loss_fn(predictions, ratings)
      loss.backward()
      optimizer.step()
      model.zero_grad()
    
    train_loss = evaluate_model(train)
    val_loss = evaluate_model(val)

    # Calling .item() releases the copy of the computational graph stored in loss
    train_losses[i] = train_loss.item()
    val_losses[i] = val_loss.item()

    print("============== EPOCH {} ==============\nTrain loss = {}\Validation loss = {}\n"\
      .format(i + 1, train_loss, val_loss))
    
    result_path = "../results/content_{}_{}".format(num_factors, learning_rate)
    model_path = "../models/content_{}_{}.pt".format(num_factors, learning_rate)

    np.save(result_path, [train_losses, val_losses])
    torch.save(model.state_dict(), model_path)

    


if __name__ == '__main__':
  args = sys.argv
  if len(args) >= 3:
    train_model(int(args[1]), float(args[2]))

     





