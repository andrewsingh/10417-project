import sys
import numpy as np
import torch
import lenskit
import pandas as pd
import math

# Hyperparameters
EPOCHS = 500
BATCH_SIZE = 64

# Data constants
HUNDRED_K_USERS = 943
HUNDRED_K_MOVIES = 1682
ONE_M_USERS = 6040
ONE_M_MOVIES = 3952
HUNDRED_K_DELIM = "\t"
ONE_M_DELIM = "::"
HUNDRED_K_TRAIN = "../data/ml-100k/ua.base"
HUNDRED_K_TEST = "../data/ml-100k/ua.test"

cos = torch.nn.CosineSimilarity()

class MatrixFactorization(torch.nn.Module):

  def __init__(self, num_users, num_movies, num_factors, isSparse=True):
    super().__init__()
    self.user_factors = torch.nn.Embedding(num_users, num_factors, max_norm=1, sparse=isSparse)
    self.movie_factors = torch.nn.Embedding(num_movies, num_factors, max_norm=1, sparse=isSparse)

    self.user_factors.weight.data.uniform_(-0.25, 0.25)
    self.movie_factors.weight.data.uniform_(-0.25, 0.25)

  def forward(self, users, movies):
    return (cos(self.user_factors(users), self.movie_factors(movies)) * 2) + 3



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
    #print(self.user_biases.weight)

  def forward(self, users, movies):
    prediction = ((cos(self.user_factors(users), self.movie_factors(movies)) + self.user_biases(users).squeeze(dim=1) + self.movie_biases(movies).squeeze(dim=1)) * 2) + 3 
    return torch.clamp(prediction, min=1, max=5)


def train_model(num_factors, learning_rate_f, learning_rate_b, isBiased, isCuda, experiment):

  train = np.loadtxt(fname=HUNDRED_K_TRAIN, dtype=np.dtype("int"), delimiter=HUNDRED_K_DELIM)
  test = np.loadtxt(fname=HUNDRED_K_TEST, dtype=np.dtype("int"), delimiter=HUNDRED_K_DELIM)
  
  train[:, 0] -= 1
  train[:, 1] -= 1
  np.random.shuffle(train)

  test[:, 0] -= 1
  test[:, 1] -= 1

  num_users = HUNDRED_K_USERS
  num_movies = HUNDRED_K_MOVIES

  if isCuda:
    if isBiased:
      model = BiasedMatrixFactorization(num_users, num_movies, num_factors).cuda()
    else:
      model = MatrixFactorization(num_users, num_movies, num_factors).cuda()
    long_type = torch.cuda.LongTensor
    float_type = torch.cuda.FloatTensor
  else:
    if isBiased:
      model = BiasedMatrixFactorization(num_users, num_movies, num_factors)
    else:
      model = MatrixFactorization(num_users, num_movies, num_factors)
    long_type = torch.LongTensor
    float_type = torch.FloatTensor

  loss_fn = torch.nn.MSELoss()
  factors_optimizer = torch.optim.SGD([model.user_factors.weight, model.movie_factors.weight], lr=learning_rate_f)
  if isBiased:
    biases_optimizer = torch.optim.SGD([model.user_biases.weight, model.movie_biases.weight], lr=learning_rate_b)

  def evaluate_model(data):
    users = torch.Tensor(data[:, 0]).type(long_type)
    movies = torch.Tensor(data[:, 1]).type(long_type)
    ratings = torch.Tensor(data[:, 2]).type(float_type)
    predictions = model(users, movies)
    return torch.sqrt(loss_fn(predictions, ratings))


  train_losses = np.zeros(EPOCHS)
  test_losses = np.zeros(EPOCHS)

  for i in range(EPOCHS):
    n = 0
    for j in range(0, math.ceil(train.shape[0] / BATCH_SIZE)):
    #for [user, movie, rating, _] in train:
      batch = train[j * BATCH_SIZE : (j + 1) * BATCH_SIZE]
      users = torch.Tensor(batch[:, 0]).type(long_type)
      movies = torch.Tensor(batch[:, 1]).type(long_type)
      ratings = torch.Tensor(batch[:, 2]).type(float_type)
      predictions = model(users, movies)
      loss = loss_fn(predictions, ratings)
      loss.backward()
      factors_optimizer.step()
      if isBiased:
        biases_optimizer.step()
      model.zero_grad()
      
      # if n % 500 == 0:
      #   print(n)
      #   print("Loss: {}".format(loss.item()))
      #   # print("User embedding: {}".format(model.user_factors(users)))
      #   # print("Movie embedding: {}".format(model.movie_factors(movies)))
      #   # if isBiased:
      #   #   print("User bias: {}".format(model.user_biases(users)))
      #   #   print("Movie bias: {}".format(model.movie_biases(movies)))
      #   print("Prediction: {}".format(predictions[0]))
      #   print("Rating: {}".format(ratings[0]))
      #   print("Diff: {}\n".format(abs(predictions[0] - ratings[0])))
      # n += 1
      
    train_loss = evaluate_model(train)
    test_loss = evaluate_model(test)

    # Calling .item() releases the copy of the computational graph stored in loss
    train_losses[i] = train_loss.item()
    test_losses[i] = test_loss.item()

    print("============== EPOCH {} ==============\nTrain loss = {}\nTest loss = {}\n"\
      .format(i + 1, train_loss, test_loss))
    

    #output_file = "results/{}".format(experiment)
    output_file = "../results/tests/{}-{}-{}-{}".format(num_factors, learning_rate_f, isBiased, experiment)

    np.save(output_file, [train_losses, test_losses])





if __name__ == '__main__':
  args = sys.argv
  if len(args) >= 7:
    train_model(int(args[1]), float(args[2]), float(args[3]), (args[4] == "y"), (args[5] == "y"), args[6])

     





