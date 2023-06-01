import argparse
import pandas as pd
from loader import Loader
from evaluater import Evaluater

def main():
        # parse arguments
        parser = argparse.ArgumentParser()
        parser.add_argument("--task", type=str, required=True)
        parser.add_argument("--lang", type=str, required=True)
        parser.add_argument("--shot", type=int, required=True)
        args = parser.parse_args()

        # load model and data based on task and language and shot
        loader = Loader(args.task, args.lang, args.shot)
        model, export_path = loader.load()

        # # when shot is 192, we evaluate all languages
        # if args.shot == 192:
        #     for lang in ["ar", "fr", "es", "ru"]:
        #         predictions = model.evaluate(lang)
        #         pd.DataFrame({"prediction": predictions}).to_csv(
        #             f"result/XVNLI/{lang}/prediction_192_shot.csv", index=False
        #         )
        #         evaluater = Evaluater(args.task, lang)
        #         accuracy = evaluater.accuracy(predictions)
        #         print("task:", args.task, "language:", lang, "accuracy:", accuracy)
        #     return
        # predict and export
        predictions = model.evaluate(args.lang)
        pd.DataFrame({"prediction": predictions}).to_csv(export_path, index=False)

        # evaluate
        evaluater = Evaluater(args.task, args.lang)
        accuracy = evaluater.accuracy(predictions)
        print("task:", args.task, "language:", args.lang, "accuracy:", accuracy)

if __name__ == "__main__":
    main()
