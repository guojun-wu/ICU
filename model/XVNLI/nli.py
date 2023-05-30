# fine tune nli

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset, RandomSampler
from transformers import get_linear_schedule_with_warmup
import pandas as pd
import numpy as np
import os


class NLI:
    def __init__(self, shot, lang):
        self.shot = shot
        self.lang = lang
        self.device = (
            torch.device("mps")
            if torch.backends.mps.is_available()
            else torch.device("cpu")
        )
        model_path = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.num_epochs = 3

    def load_data(self, data_path, batch_size):
        df = pd.read_csv(data_path, header=0)
        df = df[["caption", "hypothesis", "label"]]

        # encode data
        encoded_data = self.tokenizer.batch_encode_plus(
            df[["caption", "hypothesis"]].values.tolist(),
            add_special_tokens=True,
            return_attention_mask=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt",
        )

        # encode labels
        label_map = {"entailment": 0, "neutral": 1, "contradiction": 2}
        df["label"] = df["label"].map(label_map)

        # create dataset
        input_ids = encoded_data["input_ids"]
        attention_masks = encoded_data["attention_mask"]
        labels = torch.tensor(df["label"].values)

        dataset = TensorDataset(input_ids, attention_masks, labels)
        dataloader = DataLoader(
            dataset,
            sampler=RandomSampler(dataset),
            batch_size=batch_size,
        )
        return dataloader

    def train(self):
        # Set random seed
        seed = 42
        torch.manual_seed(seed)

        self.model.to(self.device)

        # Set model to training mode
        self.model.train()

        # define optimizer and loss function
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=1e-4,
            eps=1e-6,
            weight_decay=0.01,
            betas=(0.9, 0.999),
        )

        # create data loader
        dataloader = self.load_data(
            f"data/XVNLI/{self.lang}/train_{self.shot}.csv", batch_size=8
        )

        # setup scheduler
        total_steps = len(dataloader) * self.num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps,
        )

        # training loop
        for epoch in range(self.num_epochs):
            print(f"Epoch {epoch + 1}/{self.num_epochs}")
            print("-" * 10)

            for batch in dataloader:
                # unpack batch
                b_input_ids = batch[0].to(self.device)
                b_input_mask = batch[1].to(self.device)
                b_labels = batch[2].to(self.device)

                # clear gradients
                self.model.zero_grad()

                # forward pass
                outputs = self.model(
                    b_input_ids,
                    token_type_ids=None,
                    attention_mask=b_input_mask,
                    labels=b_labels,
                )

                # get loss
                loss = outputs[0]

                # backward pass
                loss.backward()

                max_gradient_norm = 1.0  # max gradient norm

                # clip gradients
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_gradient_norm
                )

                # update parameters
                optimizer.step()

                # update learning rate
                scheduler.step()

        # save model
        self.model.save_pretrained("model/XVNLI/fine_tune_nli")
        self.tokenizer.save_pretrained("model/XVNLI/fine_tune_nli")

    def evaluate(self):
        trained_model = self.model.to(self.device)
        if self.shot > 0:
            trained_model.load_state_dict(
                torch.load("model/XVNLI/fine_tune_nli/pytorch_model.bin")
            )
        trained_model.eval()

        predictions, true_vals = [], []

        # load test data
        dataloader = self.load_data(f"data/XVNLI/{self.lang}/test.csv", batch_size=32)

        # test loop
        for batch in dataloader:
            batch = tuple(t.to(self.device) for t in batch)

            # unpack batch
            b_input_ids, b_input_mask, b_labels = batch

            # forward pass
            with torch.no_grad():
                outputs = trained_model(
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

        # calculate accuracy
        # convert to label ids
        predictions = np.array([np.argmax(pred) for pred in predictions])
        accuracy = np.mean(predictions == true_vals).__float__()
        print(f"{self.lang} Accuracy: {accuracy}")

        # convert to labels
        label_map = {0: "entailment", 1: "neutral", 2: "contradiction"}
        predictions = [label_map[pred] for pred in predictions]

        # return predictions
        return predictions
