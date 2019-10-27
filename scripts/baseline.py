import sys
import numpy as np
import torch


# Hyperparameters
EPOCHS = 100
LEARNING_RATE = 1e-6
NUM_FACTORS = 20

# Data constants
HUNDRED_K_USERS = 943
HUNDRED_K_MOVIES = 1682
ONE_M_USERS = 6040
ONE_M_MOVIES = 3952
HUNDRED_K_DELIM = "\t"
ONE_M_DELIM = "::"
HUNDRED_K_TRAIN = "data/ml-100k/ua.base"
HUNDRED_K_TEST = "data/ml-100k/ua.test"


cos = torch.nn.CosineSimilarity()


class MatrixFactorization(torch.nn.Module):

  def __init__(self, num_users, num_movies, num_factors):
    super().__init__()
    self.user_factors = torch.nn.Embedding(num_users, NUM_FACTORS, sparse=True)
    self.movie_factors = torch.nn.Embedding(num_movies, NUM_FACTORS, sparse=True)


  def forward(self, users, movies):
    return cos(self.user_factors(users), self.movie_factors(movies))



def train_model(model, train, test, loss_fn, optimizer, long_type, float_type, experiment):
  
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
    for [user, movie, rating, _] in train:
      users = torch.Tensor([user]).type(long_type)
      movies = torch.Tensor([movie]).type(long_type)
      ratings = torch.Tensor([rating]).type(float_type)
      predictions = model(users, movies)
      loss = loss_fn(predictions, ratings)
      loss.backward()
      optimizer.step()

      if n % 100 == 0:
        print(n)
      n += 1
      
    train_loss = evaluate_model(train)
    test_loss = evaluate_model(test)

    train_losses[i] = train_loss
    test_losses[i] = test_loss

    print("\n============== EPOCH {} ==============\nTrain loss = {}\nTest loss = {}\n"\
      .format(i + 1, train_loss, test_loss))
    
    np.save("results/{}".format(experiment), [train_losses, test_losses])




def run_experiment(experiment, isCuda):
  train = np.loadtxt(fname=HUNDRED_K_TRAIN, dtype=np.dtype("int"), delimiter=HUNDRED_K_DELIM)
  test = np.loadtxt(fname=HUNDRED_K_TEST, dtype=np.dtype("int"), delimiter=HUNDRED_K_DELIM)
  
  train[:, 0] -= 1
  train[:, 1] -= 1
  np.random.shuffle(train)

  test[:, 0] -= 1
  test[:, 1] -= 1
  np.random.shuffle(test)

  num_users = HUNDRED_K_USERS
  num_movies = HUNDRED_K_MOVIES
  num_ratings = data.shape[0]
  
  if isCuda:
    model = MatrixFactorization(num_users, num_movies, NUM_FACTORS).cuda()
    long_type = torch.cuda.LongTensor
    float_type = torch.cuda.FloatTensor
  else:
    model = MatrixFactorization(num_users, num_movies, NUM_FACTORS)
    long_type = torch.LongTensor
    float_type = torch.FloatTensor

  loss_fn = torch.nn.MSELoss()
  optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

  train_model(model, train, test, loss_fn, optimizer, long_type, float_type, experiment)




if __name__ == '__main__':
  args = sys.argv
  if len(args) >= 3:
    experiment = args[1]
    isCuda = (args[2] == "y")
    run_experiment(experiment, isCuda)

     





