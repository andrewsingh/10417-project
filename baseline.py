import numpy as np
import torch


EPOCHS = 100
BATCH_SIZE = 16
NUM_FACTORS = 20
HUNDRED_K_USERS = 943
HUNDRED_K_MOVIES = 1682
ONE_M_USERS = 6040
ONE_M_MOVIES = 3952
HUNDRED_K_DELIM = "\t"
ONE_M_DELIM = "::"
HUNDRED_K_FILE = "ml-100k/u.data"
ONE_M_FILE = "ml-1m/ratings.dat"


data = np.loadtxt(fname=HUNDRED_K_FILE, dtype=np.dtype("int"), delimiter=HUNDRED_K_DELIM)
data[:, 0] -= 1
data[:, 1] -= 1
np.random.shuffle(data)

num_users = HUNDRED_K_USERS
num_movies = HUNDRED_K_MOVIES
num_ratings = data.shape[0]
num_batches = int(num_ratings / BATCH_SIZE)
cos = torch.nn.CosineSimilarity()

class MatrixFactorization(torch.nn.Module):

	def __init__(self, num_users, num_movies, num_factors):
		super().__init__()
		self.user_factors = torch.nn.Embedding(num_users, NUM_FACTORS, sparse=True)
		self.movie_factors = torch.nn.Embedding(num_movies, NUM_FACTORS, sparse=True)


	def forward(self, users, movies):
		return cos(self.user_factors(users), self.movie_factors(movies))



model = MatrixFactorization(num_users, num_movies, NUM_FACTORS)

loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-6)



def train_model(epochs):
	for i in range(epochs):
		n = 0
		for [user, movie, rating, _] in data:
			users = torch.LongTensor([user])
			movies = torch.LongTensor([movie])
			ratings = torch.FloatTensor([rating])
			predictions = model(users, movies)
			loss = loss_fn(predictions, ratings)
			loss.backward()
			optimizer.step()
			if (n % 100 == 0):
				print("{}".format(n))
			n += 1



def evaluate_model():
	users = torch.LongTensor(data[:, 0])
	movies = torch.LongTensor(data[:, 1])
	ratings = torch.FloatTensor(data[:, 2])
	print(ratings)
	predictions = model(users, movies)
	return torch.sqrt(loss_fn(predictions, ratings))




# def batch_data(ratings):
# 	ratings = np.copy(ratings)
# 	batches = []
# 	while ratings.shape[0] > 0:
# 		mask = np.full(ratings.shape[0], True, dtype=bool)
# 		users_added = np.full(num_users, False, dtype=bool)
# 		movies_added = np.full(num_movies, False, dtype=bool)
# 		batch = []
# 		for i in range(ratings.shape[0]):
# 			if not users_added[ratings[i][0]] and not movies_added[ratings[i][1]]:
# 				batch.append(ratings[i])
# 				users_added[ratings[i][0]] = True
# 				movies_added[ratings[i][1]] = True
# 				mask[i] = False
# 		batches.append(np.asarray(batch))
# 		ratings = ratings[mask]
# 		print("Batch {} added, size: {}, ratings remaining: {}".format(len(batches), len(batch), ratings.shape[0]))
# 	return np.asarray(batches)



# def train_model_batches(epochs):
# 	for i in range(epochs):
# 		n = 0
		
# 		for batch in batches:
# 			users = torch.LongTensor(batch[:, 0])
# 			movies = torch.LongTensor(batch[:, 1])
# 			ratings = torch.FloatTensor(batch[:, 2])
# 			predictions = model(users, movies)
# 			losses = loss_fn(predictions, ratings)
# 			losses.backward()
# 			optimizer.step()
# 			if (n % 100 == 0):
# 				print("{}: {}".format(i, loss))
# 			n += 1






