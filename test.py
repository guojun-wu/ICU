# test model output

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

model_path = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

label = "neutral"

# encode premise and hypothesis
encoded_input = tokenizer(
    "This is a test sentence.",
    "This is a test sentence.",
    return_tensors="pt",
    truncation=True,
)

# get prediction
with torch.no_grad():
    outputs = model(**encoded_input)

# convert probability to label
prediction = torch.softmax(outputs["logits"][0], -1).tolist()
print(prediction)


# convert label to one-hot vector, e.g. [0, 1, 0]
label_to_index = {"entailment": 0, "neutral": 1, "contradiction": 2}

numerical_label = label_to_index[label]

target = F.one_hot(torch.tensor([numerical_label]), num_classes=3).squeeze()

print(target)
print(outputs["logits"][0])

# calculate loss
loss = torch.nn.CrossEntropyLoss()
print(loss(torch.tensor(prediction), target.to(torch.float3)))
print(loss(outputs["logits"][0], target.to(torch.float32)))
