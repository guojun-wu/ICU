# Purpose: load model and data based on task and language
from model.nli import MR
from model.XVNLI.zero_shot import ZERO_SHOT_NLI
import pandas as pd


class Loader(object):
    def __init__(self, task, lang):
        self.task = task
        self.lang = lang

    def load(self):
        if self.task == "nli":
            model = ZERO_SHOT_NLI()
            df = pd.read_csv("data/xvNLI/" + self.lang + "/test.csv", sep=",", header=0)
            data = df[["label", "caption", "hypothesis"]]
            export_path = "data/xvNLI/" + self.lang + "/prediction_0shot.csv"

        elif self.task == "mr":
            model = MR()
            df = pd.read_csv(
                "data/MaRVL/" + self.lang + "/reframed.csv", sep=";", header=0
            )
            data = df[["label", "description", "caption"]]
            export_path = "data/MaRVL/" + self.lang + "/prediction.csv"

        else:
            raise ValueError("Task not supported")

        return model, data, export_path
