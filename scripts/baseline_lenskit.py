import pandas as pd
import pickle
from lenskit.algorithms.als import BiasedMF
from lenskit.batch import predict
from lenskit.metrics.predict import rmse

train = pd.read_pickle("../data/ml-20m-split/train.pkl")
val = pd.read_pickle("../data/ml-20m-split/val.pkl")
test = pd.read_pickle("../data/ml-20m-split/test.pkl")


model = BiasedMF(30, iterations=100)
print("Fitting model...")
model.fit(train)
print("Making predictions...")
preds = predict(model, val)
result = rmse(preds["prediction"], preds["rating"])

print("============= RESULTS =============\nFactors: {}\nIterations: {}\nRMSE: {}"\
    .format(30, 100, result))


