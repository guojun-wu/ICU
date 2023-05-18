import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import os


class FEW_SHOT_NLI:
    def __init__(self, shot, lang):
        self.shot = shot
        self.lang = lang
        self.label = ["entailment", "neutral", "contradiction"]
        self.device = (
            torch.device("mps")
            if torch.backends.mps.is_available()
            else torch.device("cpu")
        )
        model_path = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.data = self.load_data()
        self.num_epochs = 5
        self.train()

    def load_data(self):
        # load few-shot data
        # only shot 1, 5, 10, 20, 25, 48 are available
        if self.shot not in [1, 5, 10, 20, 25, 48]:
            raise ValueError("Shot number not supported")

        data = pd.read_csv(
            f"data/XVNLI/{self.lang}/train_{self.shot}.csv",
            header=0,
        )

        data = data[["caption", "hypothesis", "label"]]
        return data

    def train(self):
        # define optimizer and loss function
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.00001)
        loss_fn = torch.nn.CrossEntropyLoss()

        class_to_idx = {"entailment": 0, "neutral": 1, "contradiction": 2}

        # training loop for few-shot learning
        for epoch in range(self.num_epochs):
            for premise, hypothesis, label in self.data.values:
                # zero the parameter gradients
                optimizer.zero_grad()

                # encode premise and hypothesis
                encoded_input = self.tokenizer(
                    premise, hypothesis, return_tensors="pt", truncation=True
                ).to(self.device)

                # get prediction
                outputs = self.model(**encoded_input)

                prediction = torch.softmax(outputs["logits"][0], -1)

                # convert label to index
                numerical_label = class_to_idx[label]

                target = (
                    F.one_hot(torch.tensor([numerical_label]), num_classes=3)
                    .squeeze()
                    .to(self.device)
                )

                # calculate loss
                loss = loss_fn(
                    prediction,
                    target.to(torch.float32),
                )

                # backward pass and optimization
                loss.backward()
                optimizer.step()

        # print training finished
        print(f"Language {self.lang} Training finished for {self.shot}-shot model")

        # Save the model's state dict
        torch.save(
            self.model.state_dict(),
            f"model/XVNLI/{self.lang}_{self.shot}_shot_checkpoint.pth",
        )

    def predict(self, data):
        # Load trained model
        trained_model = self.model.to(self.device)
        trained_model.load_state_dict(
            torch.load(
                f"model/XVNLI/{self.lang}_{self.shot}_shot_checkpoint.pth",
                map_location=self.device,
            )
        )
        trained_model.eval()

        predictions = []
        for premise, hypothesis in data.values:
            # Encode premise and hypothesis
            encoded_input = self.tokenizer(
                premise, hypothesis, return_tensors="pt", truncation=True
            ).to(self.device)

            # Get prediction
            with torch.no_grad():
                outputs = trained_model(**encoded_input)

            # Convert logits to predicted label
            prediction = torch.softmax(outputs["logits"][0], -1).tolist()
            prediction = self.label[prediction.index(max(prediction))]
            predictions.append(prediction)

        # rmove the trained model from device
        os.remove(f"model/XVNLI/{self.lang}_{self.shot}_shot_checkpoint.pth")
        return predictions
