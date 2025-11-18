import pickle

with open("feinstaubdataexercise.pickle", "rb") as f:
    data = pickle.load(f)

print(type(data))
print(len(data))
print(data)
