import sys
import numpy as np
import torch
import pandas as pd
import pickle
import time


EPOCHS = 500
is_cuda = False
verbose = True


# Load and preprocess data
train = pd.read_pickle("../data/splits/train.pkl")
val = pd.read_pickle("../data/splits/val.pkl")

train["user"] -= 1
val["user"] -= 1
train = train.sample(frac=1) # Shuffle the training set

with open("../data/item_dict.pkl", "rb") as f:
  movie_dict = pickle.load(f)

# Data constants
NUM_USERS = len(train.groupby("user").size())
NUM_MOVIES = len(movie_dict)


cos = torch.nn.CosineSimilarity()


class MatrixFactorization(torch.nn.Module):

  def __init__(self, num_users, num_movies, num_factors):
    super().__init__()
    self.user_factors = torch.nn.Embedding(num_users, num_factors, max_norm=1, sparse=True)
    self.movie_factors = torch.nn.Embedding(num_movies, num_factors, max_norm=1, sparse=True)

    self.user_factors.weight.data.uniform_(-0.25, 0.25)
    self.movie_factors.weight.data.uniform_(-0.25, 0.25)

  def forward(self, users, movies):
    return (cos(self.user_factors(users), self.movie_factors(movies)) * 2.25) + 2.75



class BiasedMatrixFactorization(torch.nn.Module):

  def __init__(self, num_users, num_movies, num_factors):
    super().__init__()
    self.user_factors = torch.nn.Embedding(num_users, num_factors, max_norm=1, sparse=True)
    self.movie_factors = torch.nn.Embedding(num_movies, num_factors, max_norm=1, sparse=True)
    self.user_biases = torch.nn.Embedding(num_users, 1, max_norm=2, sparse=True)
    self.movie_biases = torch.nn.Embedding(num_movies, 1, max_norm=2, sparse=True)

    self.user_factors.weight.data.uniform_(-0.25, 0.25)
    self.movie_factors.weight.data.uniform_(-0.25, 0.25)
    self.user_biases.weight.data.uniform_(-0.25, 0.25)
    self.movie_biases.weight.data.uniform_(-0.25, 0.25)

  def forward(self, users, movies):
    prediction = ((cos(self.user_factors(users), self.movie_factors(movies)) + \
      self.user_biases(users).squeeze(dim=1) + self.movie_biases(movies).squeeze(dim=1)) * 2.25) + 2.75 
    return torch.clamp(prediction, min=0.5, max=5)



def train_model(num_factors, learning_rate, isBiased):

  if is_cuda:
    if isBiased:
      model = BiasedMatrixFactorization(NUM_USERS, NUM_MOVIES, num_factors).cuda()
    else:
      model = MatrixFactorization(NUM_USERS, NUM_MOVIES, num_factors).cuda()
    long_type = torch.cuda.LongTensor
    float_type = torch.cuda.FloatTensor
    torch.backends.cudnn.benchmark=True
  else:
    if isBiased:
      model = BiasedMatrixFactorization(NUM_USERS, NUM_MOVIES, num_factors)
    else:
      model = MatrixFactorization(NUM_USERS, NUM_MOVIES, num_factors)
    long_type = torch.LongTensor
    float_type = torch.FloatTensor

  loss_fn = torch.nn.MSELoss()
  optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

  def evaluate_model(df):
    movie_indices = [movie_dict[mid] for mid in df["item"].values]

    users = torch.Tensor(df["user"].values).type(long_type)
    movies = torch.Tensor(movie_indices).type(long_type)
    ratings = torch.Tensor(df["rating"].values).type(float_type)
    predictions = model(users, movies)
    return torch.sqrt(loss_fn(predictions, ratings))


  train_losses = np.zeros(EPOCHS)
  val_losses = np.zeros(EPOCHS)

  for i in range(EPOCHS):
    n = 0
    # time1 = time.time()
    # time2 = time1
    for row in train.itertuples(index=False):
      # time1 = time.time()
      # print("Time Elapsed 1: {}".format(time1 - time2))

      users = torch.Tensor([row.user]).type(long_type)
      movies = torch.Tensor([movie_dict[row.item]]).type(long_type)
      ratings = torch.Tensor([row.rating]).type(float_type)
      predictions = model(users, movies)
      loss = loss_fn(predictions, ratings)
      loss.backward()
      optimizer.step()
      model.zero_grad()
      
      # if verbose and (n % 10000 == 0):
      #   print(n)
      #   print("Loss: {}".format(loss.item()))
      #   print("User embedding: {}".format(model.user_factors(users)))
      #   print("Movie embedding: {}".format(model.movie_factors(movies)))
      #   if isBiased:
      #     print("User bias: {}".format(model.user_biases(users)))
      #     print("Movie bias: {}".format(model.movie_biases(movies)))
      #   print("Prediction: {}".format(predictions[0]))
      #   print("Rating: {}".format(ratings[0]))
      #   print("Diff: {}\n".format(abs(predictions[0] - ratings[0])))
      # n += 1

      # time2 = time.time()
      # print("Time Elapsed 2: {}".format(time2 - time1))
      
    train_loss = evaluate_model(train)
    val_loss = evaluate_model(val)

    # Calling .item() releases the copy of the computational graph stored in loss
    train_losses[i] = train_loss.item()
    val_losses[i] = val_loss.item()

    print("============== EPOCH {} ==============\nTrain loss = {}\Validation loss = {}\n"\
      .format(i + 1, train_loss, val_loss))
    
    result_path = "../results/{}-{}-{}".format(num_factors, learning_rate, isBiased)
    model_path = "../models/{}-{}-{}.pt".format(num_factors, learning_rate, isBiased)

    np.save(result_path, [train_losses, val_losses])
    torch.save(model.state_dict(), model_path)

    





if __name__ == '__main__':
  args = sys.argv
  if len(args) >= 4:
    train_model(int(args[1]), float(args[2]), (args[3] == "y"))

     





