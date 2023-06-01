import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset, RandomSampler
from transformers import get_linear_schedule_with_warmup
import pandas as pd
import numpy as np



class NLR:
    def __init__(self, lang):
        self.lang = lang
        self.device = (
            torch.device("mps")
            if torch.backends.mps.is_available()
            else torch.device("cpu")
        )
        model_path = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)

    def load_data(self, data_path, batch_size, shuffle=True):
        df = pd.read_csv(data_path, sep=';', header=0)
        df = df[["generated_caption", "original_caption", "label"]]

        # encode data
        encoded_data = self.tokenizer.batch_encode_plus(
            df[["generated_caption", "original_caption"]].values.tolist(),
            add_special_tokens=True,
            return_attention_mask=True,
            padding="max_length",
            max_length=256,
            return_tensors="pt",
            truncation=True,
        )

        # encode labels
        label_map = {True: 0, False: 1}
        df["label"] = df["label"].map(label_map)

        # create dataset
        input_ids = encoded_data["input_ids"]
        attention_masks = encoded_data["attention_mask"]
        labels = torch.tensor(df["label"].values)

        dataset = TensorDataset(input_ids, attention_masks, labels)

        if not shuffle:
            dataloader = DataLoader(
                dataset,
                sampler=None,
                batch_size=batch_size,
            )
            return dataloader
        
        dataloader = DataLoader(
            dataset,
            sampler=RandomSampler(dataset),
            batch_size=batch_size,
        )
        return dataloader
             

    def evaluate(self, lang):
        self.model.to(self.device)
        self.model.eval()
 

        predictions, true_vals = [], []

        # load test data
        dataloader = self.load_data(f"data/MaRVL/{lang}/test.csv", batch_size=32, shuffle=False)

        # test loop
        for batch in dataloader:
            batch = tuple(t.to(self.device) for t in batch)

            # unpack batch
            b_input_ids, b_input_mask, b_labels = batch

            # forward pass
            with torch.no_grad():
                outputs = self.model(
                    b_input_ids,
                    token_type_ids=None,
                    attention_mask=b_input_mask,
                )

            # get predictions
            logits = outputs[0]
            prediction = logits.detach().cpu().numpy()
            predictions.extend(prediction)

            # get true values
            label_ids = b_labels.to("cpu").numpy()
            true_vals.extend(label_ids)

        # convert to label ids, if predition[0] > prediction[2] -> 0 else 1
        predictions = [0 if pred[0] > pred[2] else 1 for pred in predictions]

        # convert to labels
        label_map = {0: "True", 1: "False"}
        predictions = [label_map[pred] for pred in predictions]

        # return predictions
        return predictions