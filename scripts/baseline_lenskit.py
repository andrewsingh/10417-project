import pandas as pd
import pickle
from lenskit.algorithms.als import BiasedMF
from lenskit.batch import predict
from lenskit.metrics.predict import rmse

train = pd.read_pickle("../data/splits/train.pkl")
val = pd.read_pickle("../data/splits/val.pkl")
test = pd.read_pickle("../data/splits/test.pkl")


algo = BiasedMF(30)
algo.fit(train[:600])
preds = predict(algo, val[100:300])
preds["prediction"].values
rmse(preds["prediction"], preds["rating"])
val[200:400]


