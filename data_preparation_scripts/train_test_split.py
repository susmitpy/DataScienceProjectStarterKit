import pandas as pd


df = pd.read_csv("../data/data.csv")
train = df.sample(frac=0.8)
test = df.drop(train.index)

train.to_csv("../data/train.csv",index=False) # Find patterns on train
test.to_csv("../data/test.csv",index=False) # Validate assumptions found on test

