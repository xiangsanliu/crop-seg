import os
import pandas as pd


root_dir = "dataset/tianchi_gan"

path = os.path.join(root_dir, "label")

filenames = os.listdir(path)

filename = pd.DataFrame(filenames)

filename.to_csv(os.path.join(root_dir, "a.csv"), index=False, header=None)
print(filenames)