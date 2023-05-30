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
model.deberta.encoder.layer[-1].output.dropout.p = 0.1

# get prediction
with torch.no_grad():
    outputs = model(**encoded_input)

# convert probability to label
prediction = torch.softmax(outputs["logits"][0], -1)
print(prediction)


# convert label to one-hot vector, e.g. [0, 1, 0]
label_to_index = {"entailment": 0, "neutral": 1, "contradiction": 2}

numerical_label = label_to_index[label]

target = F.one_hot(torch.tensor([numerical_label]), num_classes=3).squeeze()

print(target)
print(outputs["logits"][0])

# calculate loss
loss = torch.nn.CrossEntropyLoss()
print(loss(prediction, target.to(torch.float32)))
print(loss(outputs["logits"][0], target.to(torch.float32)))

epsilon = 0.1  # Label smoothing factor

# Create the smoothed target distribution
num_classes = 3  # Number of classes
smoothed_labels = (1 - epsilon) * torch.eye(num_classes)[
    numerical_label
] + epsilon / num_classes


print(smoothed_labels)

# Calculate the loss using the smoothed labels
print(loss(prediction, smoothed_labels.to(torch.float32)))

# Language es Training finished for 0-shot model
# 0.6104347826086957

# Language es Training finished for 1-shot model
# 0.6043478260869565

# Language en Training finished for 5-shot model
# 0.6078260869565217

# Language es Training finished for 10-shot model 2 epoch
# 0.6026086956521739

# Language es Training finished for 20-shot model 2 epoch
# 0.6208695652173913

# Language es Training finished for 20-shot model 3 epoch
# 0.6121739130434782

# Language es Training finished for 25-shot model 2 epoch
# 0.6034782608695652

# Language es Training finished for 25-shot model 3 epoch
# 0.6026086956521739

# Language es Training finished for 48-shot model 1 epoch
# 0.6086956521739131
# Language es Training finished for 48-shot model 2 epoch
# 0.611304347826087
# Language es Training finished for 48-shot model 3 epoch
# 0.611304347826087
