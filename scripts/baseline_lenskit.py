import pandas as pd
import pickle
from lenskit.algorithms.als import BiasedMF
from lenskit.batch import predict
from lenskit.metrics.predict import rmse


test_bool = True

train = pd.read_pickle("../data/ml-1m-split/train.pkl")
val = pd.read_pickle("../data/ml-1m-split/val.pkl")
test = pd.read_pickle("../data/ml-1m-split/test.pkl")

num_factors = 30
num_iters = 100

model = BiasedMF(num_factors, iterations=num_iters)
print("Fitting model...")
model.fit(train)
print("Making validation predictions...")
val_preds = predict(model, val)
val_result = rmse(val_preds["prediction"], val_preds["rating"])

if test_bool:
    print("Making test predictions...")
    test_preds = predict(model, test)
    test_result = rmse(test_preds["prediction"], test_preds["rating"])
else:
    test_result = 0

print("============= RESULTS =============\nFactors: {}\nIterations: {}\nValidation RMSE: {}\nTest RMSE: {}" \
    .format(num_factors, num_iters, val_result, test_result))



