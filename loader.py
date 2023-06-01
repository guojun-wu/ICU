from model.MaRVL.nlr import NLR
from model.XVNLI.nli import NLI
import pandas as pd
from data.MaRVL.dataset import MaRVL_Dataset_Generator


class Loader(object):
    def __init__(self, task, lang, shot_or_frame):
        self.task = task
        self.lang = lang
        self.shot_or_frame = shot_or_frame

    def load(self):
        if self.task == "nli":
            # only support language in ["ar", "fr", "es", "ru"]
            if self.lang not in ["ar", "fr", "es", "ru"]:
                raise ValueError("Language not supported")
            
            shot = self.shot_or_frame
            
            # only support shot in [0, 1, 5, 10, 20, 25, 48, 192]
            if shot not in [0, 1, 5, 10, 20, 25, 48, 192]:
                raise ValueError("Shot not supported")

            model = NLI(shot, self.lang)
            if shot > 0:
                model.train()

            export_path = f"result/XVNLI/{self.lang}/prediction_{shot}_shot.csv"

        elif self.task == "nlr":
            # only support language in ["id", "sw", "ta", "tr", "zh"]
            if self.lang not in ["id", "sw", "ta", "tr", "zh"]:
                raise ValueError("Language not supported")
            
            frame = self.shot_or_frame

            # only support shot in [0, 1, 5, 10, 20, 25, 48]
            if frame not in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
                raise ValueError("Frame not supported")
            
            MaRVL_Dataset_Generator(frame, self.lang)

            model = NLR(self.lang)
            
            export_path = f"result/MaRVL/{self.lang}/prediction_frame_{frame}.csv"
        else:
            raise ValueError("Task not supported")

        return model, export_path
