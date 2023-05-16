import numpy as np


class Evaluater:
    def __init__(self, task):
        if task == "nli":
            self.label_dict = {"entailment": 0, "neutral": 1, "contradiction": 2}
        elif task == "mr":
            self.label_dict = {"True": 0, "False": 1}

    def evaluate(self, y_true, y_pred):
        y_true = np.array([self.label_dict[str(label)] for label in y_true])
        y_pred = np.array([self.label_dict[str(label)] for label in y_pred])
        accuracy = np.mean(y_true == y_pred).__float__()
        return accuracy
