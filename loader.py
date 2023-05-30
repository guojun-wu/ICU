from model.dual import MR
from model.XVNLI.nli import NLI
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

            model = NLI(self.shot, self.lang)
            if self.shot > 0:
                model.train()

            export_path = f"result/XVNLI/{self.lang}/prediction_{self.shot}_shot.csv"

        #elif self.task == "mr":
        else:
            raise ValueError("Task not supported")

        return model, export_path
