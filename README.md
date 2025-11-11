Task 1: Inference with a Pretrained Model

Google Colab Link:
https://colab.research.google.com/drive/1Nm5CVqpyP8v3T3U3vpkg-O8IkpaIvsW3#scrollTo=O9qWx6LvQ4m6

This script performs Extractive Question Answering using the distilbert-base-cased-distilled-squad model.

Script Highlights:

Imports AutoTokenizer and AutoModelForQuestionAnswering from transformers

Loads the pretrained model and its tokenizer

Iterates through a list of questions

Tokenizes each (question, context) pair

Runs inference to obtain start_logits and end_logits

Extracts the most likely answer span from the model output

Decodes the answer into readable text

Prints each question with its corresponding answer




