"""Transformers quickstart: sentiment analysis with the pipeline API.

Run:
    .venv/bin/python quickstart.py
"""

from transformers import pipeline

classifier = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
)

sentences = [
    "We are very happy to show you the 🤗 Transformers library.",
    "I've been waiting for a HuggingFace course my whole life.",
    "This is the worst movie I have ever watched.",
]

for sentence, result in zip(sentences, classifier(sentences)):
    print(f"{result['label']:>8}  {result['score']:.4f}  {sentence}")
