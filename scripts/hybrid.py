import sys
import numpy as np
import torch
from torch import nn
import pandas as pd
import pickle
import math
import os


EPOCHS = 100
MAX_BATCH_SIZE = 32
ITEM_HIDDIM = 150
is_cuda = True
verbose = True
hybrid = False
if hybrid:
  model_type = "hybrid2"
else:
  model_type = "content2"


if is_cuda:
    long_type = torch.cuda.LongTensor
    float_type = torch.cuda.FloatTensor
    torch.backends.cudnn.benchmark=True
else:
  long_type = torch.LongTensor
  float_type = torch.FloatTensor


# Load and preprocess data
print("Loading training and validation data...")

train_batch_eval_path = "../data/ml-1m-split/train_batched_eval_h2_1m.npy"
val_batch_path = "../data/ml-1m-split/val_batched_h2_1m.npy"

train_batches = np.load("../data/ml-1m-split/train_batched_1m.npy", allow_pickle=True)
np.random.shuffle(train_batches) # Shuffle the training batches

train_batches_eval = np.load(train_batch_eval_path, allow_pickle=True)
val_batches = np.load(val_batch_path, allow_pickle=True)

NUM_TRAIN_BATCHES = len(train_batches)
NUM_VAL_BATCHES = len(val_batches)

print("Loading item inputs...")
all_item_inputs = np.load("../data/ml-1m-add/item_inputs_1m.npy")
ITEM_INPUT_DIM = len(all_item_inputs[0])


print("Loading user items...")
all_user_items = np.load("../data/ml-1m-add/user_items_1m.npy", allow_pickle=True)


# Data constants
NUM_USERS = all_user_items.shape[0]
NUM_ITEMS = len(all_item_inputs)
GLOBAL_AVERAGE = 3.581477826039139


class MatrixFactorization(nn.Module):
  def __init__(self, num_users, num_items, num_factors):
    super().__init__()
    self.user_factors = nn.Embedding(num_users, num_factors, sparse=False)
    self.item_factors = nn.Embedding(num_items, num_factors, sparse=False)
    
  def forward(self, users, items):
    return torch.diagonal(torch.mm(self.user_factors(users), torch.transpose(self.item_factors(items), 0, 1)))
    


class ContentRecommender(nn.Module):
  def __init__(self, num_factors, item_hiddim, user_hiddim):
    super().__init__()
    self.outdim = num_factors

    # Item model
    self.item_hiddim = item_hiddim
    self.item_model = nn.Sequential(
      nn.Linear(ITEM_INPUT_DIM, self.item_hiddim),
      nn.ReLU(),
      nn.Linear(self.item_hiddim, self.item_hiddim),
      nn.ReLU(),
      nn.Linear(self.item_hiddim, self.outdim)
    )

    # User model
    self.user_hiddim = user_hiddim # num_factors
    self.user_lstm = nn.LSTM(self.outdim, self.user_hiddim, batch_first=True)


  def forward(self, user_item_inputs, item_inputs, batch_size):
    user_inputs = self.item_model(user_item_inputs).view(batch_size, -1, self.outdim)
    item_embeddings = self.item_model(item_inputs)
    (lstm_outputs, _) = self.user_lstm(user_inputs)
    user_embeddings = lstm_outputs.permute(1, 0, 2)[-1]
    assert(item_embeddings.shape == user_embeddings.shape)
    return torch.diagonal(torch.mm(user_embeddings, torch.transpose(item_embeddings, 0, 1)))



class ContentOnlyRecommender(nn.Module):
  def __init__(self, num_users, num_items, num_factors, item_hiddim, user_hiddim):
    super().__init__()

    #self.mf_model = MatrixFactorization(num_users, num_items, num_factors)
    self.content_model = ContentRecommender(num_factors, item_hiddim, user_hiddim)

    self.user_biases = nn.Embedding(num_users, 1, sparse=False)
    self.item_biases = nn.Embedding(num_items, 1, sparse=False)

    self.user_biases.weight.data.uniform_(-0.25, 0.25)
    self.item_biases.weight.data.uniform_(-0.25, 0.25)


  def forward(self, users, items, user_item_inputs, item_inputs):
    return GLOBAL_AVERAGE + self.user_biases(users).squeeze(dim=1) + self.item_biases(items).squeeze(dim=1) \
      + self.content_model(user_item_inputs, item_inputs, users.shape[0])




class HybridRecommender(nn.Module):
  def __init__(self, num_users, num_items, num_factors, item_hiddim, user_hiddim, content_path=None, biased_mf_path = None):
    super().__init__()

    self.mf_model = MatrixFactorization(num_users, num_items, num_factors)
    self.user_biases = nn.Embedding(num_users, 1, sparse=False)
    self.item_biases = nn.Embedding(num_items, 1, sparse=False)
    if biased_mf_path != None:
      with open(biased_mf_path, "rb") as f:
        biased_mf_model = pickle.load(f)
      self.mf_model.user_factors.weight = torch.nn.Parameter(torch.Tensor(biased_mf_model.user_features_).type(float_type))
      item_features = np.random.uniform(-0.25, 0.25, (NUM_ITEMS, num_factors))
      for (idx, item_feat) in zip(biased_mf_model.item_index_, biased_mf_model.item_features_):
        item_features[idx] = item_feat
      self.mf_model.item_factors.weight = torch.nn.Parameter(torch.Tensor(item_features).type(float_type))

      self.user_biases.weight = torch.nn.Parameter(torch.Tensor(biased_mf_model.user_bias_).type(float_type).view(-1, 1))
      item_biases = np.random.uniform(-0.25, 0.25, NUM_ITEMS)
      for (idx, item_bias) in zip(biased_mf_model.item_index_, biased_mf_model.item_bias_):
        item_biases[idx] = item_bias
      self.item_biases.weight = torch.nn.Parameter(torch.Tensor(item_biases).type(float_type).view(-1, 1))
    else:
      self.user_biases.weight.data.uniform_(-0.25, 0.25)
      self.item_biases.weight.data.uniform_(-0.25, 0.25)

    if content_path != None:
      content_only_model = ContentOnlyRecommender(num_users, num_items, num_factors, item_hiddim, user_hiddim)
      content_only_model.load_state_dict(torch.load(content_path))
      self.content_model = content_only_model.content_model
    else:
      self.content_model = ContentRecommender(num_factors, item_hiddim, user_hiddim)

    self.interp = nn.Parameter(torch.Tensor([0.5])).type(float_type)

    
  def forward(self, users, items, user_item_inputs, item_inputs):
    return GLOBAL_AVERAGE + self.user_biases(users).squeeze(dim=1) + self.item_biases(items).squeeze(dim=1) \
      + (self.interp * self.mf_model(users, items)) + ((1 - self.interp) * self.content_model(user_item_inputs, item_inputs, users.shape[0]))



def train_model(num_factors, learning_rate):
  print("Training {} model...".format(model_type))
  
  if is_cuda:
    if hybrid:
      content_path = "../models/content_30_0.01/epoch_23.pt"
      biased_mf_path = "../models/baseline_model.pkl"
      model = HybridRecommender(NUM_USERS, NUM_ITEMS, num_factors, ITEM_HIDDIM, num_factors, \
        content_path=content_path, biased_mf_path=biased_mf_path).cuda()
    else:
      model = ContentOnlyRecommender(NUM_USERS, NUM_ITEMS, num_factors, ITEM_HIDDIM, num_factors).cuda()   
  else:
    if hybrid:
      model = HybridRecommender(NUM_USERS, NUM_ITEMS, num_factors, ITEM_HIDDIM, num_factors, \
        content_path=content_path, biased_mf_path=biased_mf_path)
    else:
      model = ContentOnlyRecommender(NUM_USERS, NUM_ITEMS, num_factors, ITEM_HIDDIM, num_factors)
    
  loss_fn = nn.MSELoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


  def evaluate_model(batches):
    num_batches = len(batches)
    losses = []
    j = 0
    for batch in batches:
      j += 1
      users_numpy = batch[:, 0].astype(int)
      items_numpy = batch[:, 1].astype(int)
      users = torch.Tensor(users_numpy).type(long_type)
      items = torch.Tensor(items_numpy).type(long_type)
      ratings = torch.Tensor(batch[:, 2]).type(float_type)
      
      user_item_indices = np.concatenate(all_user_items[users_numpy])
      user_item_inputs = torch.Tensor(all_item_inputs[user_item_indices]).type(float_type)
      item_inputs = torch.Tensor(all_item_inputs[items_numpy]).type(float_type)

      predictions = model(users, items, user_item_inputs, item_inputs)
      loss = loss_fn(predictions, ratings)
      # Calling .item() releases the copy of the computational graph stored in loss
      losses.append(loss.item())

      if j % 100 == 0:
        print("Batch {} of {}".format(j, num_batches))
    
    return np.sqrt(np.mean(losses))

  train_losses = np.zeros(EPOCHS)
  val_losses = np.zeros(EPOCHS)

  
  for i in range(EPOCHS):
    j = 0
    for batch in train_batches[:1]:
      j += 1
      users_numpy = batch[:, 0].astype(int)
      items_numpy = batch[:, 1].astype(int)
      users = torch.Tensor(users_numpy).type(long_type)
      items = torch.Tensor(items_numpy).type(long_type)
      ratings = torch.Tensor(batch[:, 2]).type(float_type)
      
      user_item_indices = np.concatenate(all_user_items[users_numpy])
      user_item_inputs = torch.Tensor(all_item_inputs[user_item_indices]).type(float_type)
      item_inputs = torch.Tensor(all_item_inputs[items_numpy]).type(float_type)

      predictions = model(users, items, user_item_inputs, item_inputs)
      loss = loss_fn(predictions, ratings)
      loss.backward()
      optimizer.step()
      model.zero_grad()

      if j % 1000 == 0:
        print("Batch {} of {}".format(j, NUM_TRAIN_BATCHES))
        print("Loss: {}".format(loss))
    
    print("Evaluating on training set...")
    train_loss = evaluate_model(train_batches_eval)
    print("Evaluating on validation set...")
    val_loss = evaluate_model(val_batches)
    
    train_losses[i] = train_loss
    val_losses[i] = val_loss

    print("============== EPOCH {} ==============\nTrain loss = {}\nValidation loss = {}\n"\
      .format(i + 1, train_loss, val_loss))
    
    result_path = "../results/{}_{}_{}".format(model_type, num_factors, learning_rate)
    model_dir = "../models/{}_{}_{}".format(model_type, num_factors, learning_rate)
    model_path = model_dir + "/epoch_{}.pt".format(i + 1)

    if not os.path.exists(model_dir):
      os.mkdir(model_dir)

    np.save(result_path, [train_losses, val_losses])
    torch.save(model.state_dict(), model_path)

    


if __name__ == '__main__':
  args = sys.argv
  if len(args) >= 3:
    train_model(int(args[1]), float(args[2]))

     





