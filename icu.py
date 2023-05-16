import argparse
import pandas as pd
from load import Loader
from evaluater import Evaluater


class ICU:
    def main():
        # parse arguments
        parser = argparse.ArgumentParser()
        parser.add_argument("--task", type=str, required=True)
        parser.add_argument("--lang", type=str, required=True)
        args = parser.parse_args()

        # load model and data based on task and language
        loader = Loader(args.task, args.lang)
        model, data, export_path = loader.load()

        # predict and export
        prediction = model.predict(data.iloc[:, [1, 2]])
        # export prediction with id
        pd.DataFrame({"prediction": prediction}).to_csv(export_path, index=False)

        # evaluate
        evaluater = Evaluater(args.task)
        accuracy = evaluater.evaluate(data["label"], prediction)
        print(args.task, "language:", args.lang, "accuracy:", accuracy)

    if __name__ == "__main__":
        main()
