# open data/XVNLI/{lang}/train_48.csv of all languages
# put them into a single file

import pandas as pd
import numpy as np
import os

# load data
data_path = "data/XVNLI"
langs = ['ar', 'es', 'fr', 'ru']
df = pd.DataFrame()
for lang in langs:
    df = df.append(pd.read_csv(os.path.join(data_path, lang, "train_48.csv"), header=0))

# save data
df.to_csv("data/XVNLI/train_192.csv", index=False)

