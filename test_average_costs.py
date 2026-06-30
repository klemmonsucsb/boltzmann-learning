import pickle

for name in ["MLPRegressor", "BayesianEstimator"]:
    with open(f"../data/rps_40000_False/{name}.pkl", "rb") as f:
        data = pickle.load(f)
    print(name, "keys:", list(data.keys()))
    print(name, "average_costs sample:", data['average_costs'][:5], "...", data['average_costs'][1000:1005])