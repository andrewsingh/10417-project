import sys
import numpy as np
import torch
from torch import nn
import pandas as pd
import pickle
import time


EPOCHS = 500
EMBEDDING_DIM = 50
is_cuda = False
verbose = True


# Load and preprocess data
train = pd.read_pickle("../data/ml-20m-split/train.pkl")
val = pd.read_pickle("../data/ml-20m-split/val.pkl")

train = train.sample(frac=1) # Shuffle the training set


# Data constants
NUM_USERS = len(train.groupby("user").size())
NUM_ITEMS = len(item_dict)

GLOBAL_AVERAGE = train["rating"].mean()

all_item_inputs = np.load("../data/item_inputs")
all_user_items = np.load("../data/user_items")

print("Loading GloVe embeddings...")
with open("../data/glove.twitter.27B/glove.twitter.27B.{}d.txt".format(EMBEDDING_DIM)) as glove_file:
    glove_embeddings = defaultdict(lambda: np.random.rand(EMBEDDING_DIM))
    i = 0
    for line in glove_file.readlines():
        values = line.split(" ")
        word = values[0]
        embedding = np.asarray(values[1:], dtype="float32")
        assert(len(embedding) == EMBEDDING_DIM)
        glove_embeddings[word] = embedding
        
        if i % 100000 == 0:
            print(i)
        i += 1



class MatrixFactorization(nn.Module):

  def __init__(self, num_users, num_items, num_factors):
    super().__init__()
    self.user_factors = nn.Embedding(num_users, num_factors, sparse=True)
    self.item_factors = nn.Embedding(num_items, num_factors, sparse=True)
    
  def forward(self, user_indices, item_indices):
    # fix this
    return torch.mm(self.user_factors(user_indices), torch.transpose(self.item_factors(item_indices), 0, 1)).sum(-1)
    


class ContentRecommender(nn.Module):
  def __init__(self, num_factors, item_hiddim, user_hiddim):
    super().__init__()
    self.outdim = num_factors

    # Item model
    self.item_indim = 318 # don't hardcode this
    self.item_hiddim = item_hiddim # 250
    self.item_model = nn.Sequential(
      nn.Linear(self.item_indim, self.item_hiddim),
      nn.ReLU(),
      nn.Linear(self.item_hiddim, self.outdim)
    )

    # User model
    self.user_hiddim = user_hiddim # 30
    self.user_lstm = nn.LSTM(self.outdim, self.user_hiddim)


  def forward(self, user_indices, item_indices):
    user_item_indices = all_user_items[user_indices].reshape(-1)
    user_item_inputs = all_item_inputs[user_item_indices]
    user_item_inputs_indices = np.arange(user_item_inputs.shape[0]).reshape(user_indices.shape[0], -1).T.reshape(-1)
    user_inputs = self.item_model(user_item_inputs)[user_item_inputs_indices].view(-1, batch_size, self.outdim)
    item_inputs = torch.Tensor(all_item_inputs[item_indices])
    
    item_embeddings = self.item_model(item_inputs)
    (lstm_outputs, _) = self.user_lstm(user_inputs)
    user_embeddings = lstm_outputs[-1]
    assert(item_embeddings.shape == user_embeddings.shape)
    return torch.mm(user_embeddings, torch.transpose(item_embeddings, 0, 1)).sum(-1)



class HybridRecommender(nn.Module):
  def __init__(self, num_users, num_items, num_factors, item_hiddim, user_hiddim):
    super().__init__()

    self.mf_model = MatrixFactorization(num_users, num_items, num_factors)
    self.content_model = ContentRecommender(num_factors, item_hiddim, user_hiddim)

    self.user_biases = nn.Embedding(num_users, 1, sparse=True)
    self.item_biases = nn.Embedding(num_items, 1, sparse=True)

    #self.interp = nn.Embedding(num_items, 1, sparse=True)

  def forward(self, user_indices, item_ids):
    item_indices = torch.Tensor([item_dict[item_id] for item_id in item_ids])


    return GLOBAL_AVERAGE + self.user_biases(user_indices).squeeze(dim=1) + self.item_biases(item_indices).squeeze(dim=1) \
      + self.mf_model(user_indices, item_indices) + self.content_model(user_indices, item_indices)



def train_model(num_factors, learning_rate):

  if is_cuda:
    model = BiasedMF(NUM_USERS, NUM_ITEMS, num_factors).cuda()   
    long_type = torch.cuda.LongTensor
    float_type = torch.cuda.FloatTensor
    torch.backends.cudnn.benchmark=True
  else:
    model = BiasedMF(NUM_USERS, NUM_ITEMS, num_factors)
    long_type = torch.LongTensor
    float_type = torch.FloatTensor

  loss_fn = nn.MSELoss()
  optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


  def evaluate_model(df):
    item_indices = [item_dict[mid] for mid in df["item"].values]
    users = torch.Tensor(df["user"].values).type(long_type)
    items = torch.Tensor(item_indices).type(long_type)
    ratings = torch.Tensor(df["rating"].values).type(float_type)
    predictions = model(users, items)
    return torch.sqrt(loss_fn(predictions, ratings))

  train_losses = np.zeros(EPOCHS)
  val_losses = np.zeros(EPOCHS)


  for i in range(EPOCHS):
    for row in train.itertuples(index=False):
      users = torch.Tensor([row.user]).type(long_type)
      items = torch.Tensor([item_dict[row.item]]).type(long_type)
      ratings = torch.Tensor([row.rating]).type(float_type)
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
    
    result_path = "../results/augmented_{}_{}_{}".format(num_factors, learning_rate)
    model_path = "../models/augmented_{}_{}_{}.pt".format(num_factors, learning_rate)

    np.save(result_path, [train_losses, val_losses])
    torch.save(model.state_dict(), model_path)

    





if __name__ == '__main__':
  args = sys.argv
  if len(args) >= 3:
    train_model(int(args[1]), float(args[2]))

     





