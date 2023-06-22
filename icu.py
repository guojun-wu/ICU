import argparse
import pandas as pd
from loader import Loader
from evaluater import Evaluater

def main():
        # parse arguments
        parser = argparse.ArgumentParser()
        parser.add_argument("--task", type=str, required=True)
        parser.add_argument("--lang", type=str, required=True)
        parser.add_argument("--shot", type=int, required=False)
        parser.add_argument("--frame", type=int, required=False)
        
        args = parser.parse_args()

        if args.task not in ["nli", "nlr"]:
            raise ValueError("Task not supported")
        
        if args.task == "nli":
                if args.shot is None:
                        raise ValueError("Shot not specified")
                loader = Loader(args.task, args.lang, args.shot)
        else:
                if args.frame is None:
                        raise ValueError("Frame not specified")
                loader = Loader(args.task, args.lang, args.frame)

        # load model and data based on task and language and shot/frame
        model, export_path = loader.load()

        # predict
        predictions = model.evaluate(args.lang)
        pd.DataFrame({"prediction": predictions}).to_csv(export_path, index=False)

        # evaluate
        evaluater = Evaluater(args.task, args.lang)
        accuracy = evaluater.accuracy(predictions)
        print("task:", args.task, "language:", args.lang, "accuracy:", accuracy)

if __name__ == "__main__":
    main()
