from model.nli import MR
from model.XVNLI.zero_shot import ZERO_SHOT_NLI
from model.XVNLI.few_shot import FEW_SHOT_NLI
import pandas as pd


class Loader(object):
    def __init__(self, task, lang, shot):
        self.task = task
        self.lang = lang
        self.shot = shot

    def load(self):
        if self.task == "nli":
            # only support language in ["ar", "fr", "es", "ru"]
            if self.lang not in ["ar", "fr", "es", "ru"]:
                raise ValueError("Language not supported")

            if self.shot == 0:
                model = ZERO_SHOT_NLI()
                export_path = f"data/xvNLI/{self.lang}/prediction_0_shot.csv"
            elif self.shot > 0:
                model = FEW_SHOT_NLI(self.shot, self.lang)
                export_path = f"data/xvNLI/{self.lang}/prediction_{self.shot}_shot.csv"

            df = pd.read_csv(f"data/xvnli/{self.lang}/test.csv", sep=",", header=0)
            data = df[["label", "caption", "hypothesis"]]

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
