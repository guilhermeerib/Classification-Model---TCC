from flask import Flask, request, jsonify, render_template

import torch
from torch import nn
import torch.nn.functional as F

from transformers import (
    BertTokenizer,
    BertModel,
)

import torch
import matplotlib.pyplot as plt


app = Flask(__name__)


class MultiClassClassifier(nn.Module):
    def __init__(self, bert_model_name: str, hidden_size: int, num_outputs: int):
        super(MultiClassClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(0.1)
        self.linear1 = nn.Linear(self.bert.config.hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, num_outputs)

    def forward(self, input_ids, attention_mask):
        outputs_bert = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs_bert.pooler_output
        dropout = self.dropout(pooled_output)
        logits = self.linear1(dropout)
        logits = self.linear2(logits)

        return logits


BERT_MODEL_NAME = "bert-base-uncased"
hidden_size = 10
num_outputs = 2
max_length = 256

# Carregar o modelo e o tokenizador
model = MultiClassClassifier(BERT_MODEL_NAME, hidden_size, num_outputs)
model.load_state_dict(torch.load("model.pth", map_location=torch.device("cpu")))
model.eval()
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


def predict_sentiment(
    text: str,
    model: MultiClassClassifier,
    tokenizer: BertTokenizer,
    device: str,
    max_length: int,
):
    model.eval()
    encoding = tokenizer(
        text,
        return_tensors="pt",
        max_length=max_length,
        padding="max_length",
        truncation=True,
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        output = model(input_ids=input_ids, attention_mask=attention_mask)
        prob = F.softmax(output, dim=1)
    
    prob_neg = prob[0, 0].item()
    prob_pos = prob[0, 1].item()
    
    predicted_class = 'positive' if prob_pos > prob_neg else "negative"
    probability = max(prob_neg, prob_pos) * 100
    
    return predicted_class, f'{probability:.2f}%'


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def get_predict_sentiment():
    text = request.form["text"]
    prediction, probability = predict_sentiment(text, model, tokenizer, device, max_length)
    
    return render_template(
        "index.html",
        prediction=prediction,
        probability=probability,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
