# fine tune the model on the few-shot dataset

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd


class FEW_SHOT_NLI:
    def __init__(self, num_shot, model_name="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"):
        self.num_shot = num_shot
        self.device = (
            torch.device("mps")
            if torch.backends.mps.is_available()
            else torch.device("cpu")
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.label = ["entailment", "neutral", "contradiction"]
        self.model.to(self.device)
        # load few-shot data by shot_num
        self.data = self.load_data()
        self.num_epochs = 5
        self.train()

    def load_data(self):
        # load few-shot data
        data = pd.read_csv(
            f"model/XVNLI/data/{self.shot_num}_shot.csv",
            header=None,
            names=["premise", "hypothesis", "label"],
        )
        return data

    def train(self):
        # define optimizer and loss function
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        loss_fn = torch.nn.CrossEntropyLoss()

        class_to_idx = {"entailment": 0, "contradiction": 1, "neutral": 2}

        # training loop for few-shot learning
        for epoch in range(self.num_epochs):
            for premise, hypothesis, label in self.data.values:
                # zero the parameter gradients
                optimizer.zero_grad()

                # encode premise and hypothesis
                encoded_input = self.tokenizer(
                    premise, hypothesis, return_tensors="pt", truncation=True
                )
                encoded_input.to(self.device)

                # get prediction
                outputs = self.model(**encoded_input)
                prediction = torch.argmax(outputs["logits"][0], dim=-1)

                # convert label to index
                numerical_label = class_to_idx[label]

                # calculate loss
                loss = loss_fn(
                    prediction, torch.tensor(numerical_label).to(self.device)
                )

                # backward pass and optimization
                loss.backward()
                optimizer.step()

        # save model checkpoint
        torch.save(
            self.model.state_dict(),
            f"model/XVNLI/model/{self.shot_num}_shot_checkpoint.pth",
        )

    def predict(self, data):
        # load trained model
        trained_model = self.model.load_state_dict(
            torch.load(
                f"model/XVNLI/model/{self.shot_num}_shot_checkpoint.pth",
                map_location=self.device,
            )
        )

        predictions = []
        for premise, hypothesis in data.values:
            # encode premise and hypothesis
            encoded_input = self.tokenizer(
                premise, hypothesis, return_tensors="pt", truncation=True
            )
            encoded_input.to(self.device)

            # get prediction
            with torch.no_grad():
                outputs = trained_model(**encoded_input)

            # convert probability to label
            prediction = torch.softmax(outputs["logits"][0], -1).tolist()
            prediction = self.label[prediction.index(max(prediction))]
            predictions.append(prediction)
        return predictions
