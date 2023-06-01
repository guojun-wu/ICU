import numpy as np
import pandas as pd


class Evaluater:
    def __init__(self, task, lang):
        if task == "nli":
            self.label_dict = {"entailment": 0, "neutral": 1, "contradiction": 2}
            df = pd.read_csv(f"data/XVNLI/{lang}/test.csv", sep=",", header=0)
            self.labels = df["label"].values
        elif task == "nlr":
            self.label_dict = {"True": 0, "False": 1}
            df = pd.read_csv(f"data/MaRVL/{lang}/test.csv", sep=";", header=0)
            self.labels = df["label"].values
        else:
            raise ValueError("Task not supported")
        
        
    def accuracy(self, predictions):
        y_true = np.array([self.label_dict[str(label)] for label in self.labels])
        y_pred = np.array([self.label_dict[str(label)] for label in predictions])
        accuracy = np.mean(y_true == y_pred).__float__()
        return accuracy
