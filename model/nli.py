from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch


class NLI:
    def __init__(self, model_name="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"):
        self.device = (
            torch.device("mps")
            if torch.backends.mps.is_available()
            else torch.device("cpu")
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.label = ["entailment", "neutral", "contradiction"]
        self.model.to(self.device)

    def predict(self, data):
        predictions = []
        for premise, hypothesis in data.values:
            # encode premise and hypothesis
            encoded_input = self.tokenizer(
                premise, hypothesis, return_tensors="pt", truncation=True
            )
            encoded_input.to(self.device)

            # get prediction
            with torch.no_grad():
                outputs = self.model(**encoded_input)

            # convert probability to label
            prediction = torch.softmax(outputs["logits"][0], -1).tolist()
            prediction = self.label[prediction.index(max(prediction))]
            predictions.append(prediction)
        return predictions


class MR:
    def __init__(self, model_name="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"):
        self.device = (
            torch.device("mps")
            if torch.backends.mps.is_available()
            else torch.device("cpu")
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.label = ["True", "False"]
        self.model.to(self.device)

    def predict(self, data):
        predictions = []
        for premise, hypothesis in data.values:
            # encode premise and hypothesis
            encoded_input = self.tokenizer(
                premise, hypothesis, return_tensors="pt", truncation=True
            )

            encoded_input.to(self.device)

            # get prediction
            with torch.no_grad():
                outputs = self.model(**encoded_input)

            # convert prediction to label by comparing the probability
            prediction = torch.softmax(outputs["logits"][0], -1).tolist()

            if max(prediction) == prediction[0]:
                prediction = self.label[0]
            elif max(prediction) == prediction[2]:
                prediction = self.label[1]
            else:
                prediction = (
                    self.label[0] if prediction[0] > prediction[2] else self.label[1]
                )
            predictions.append(prediction)
        return predictions
