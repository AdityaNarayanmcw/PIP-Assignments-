# -*- coding: utf-8 -*-
"""pip-Interference.ipynb

google colab link: https://colab.research.google.com/drive/1Nm5CVqpyP8v3T3U3vpkg-O8IkpaIvsW3#scrollTo=O9qWx6LvQ4m6
Task 1: Inference with a Pretrained Model (Extractive Question Answering) Task Description: Write a Python script that loads a pretrained Extractive Question Answering model and its tokenizer from the Hugging Face transformers library. Your script must use this model to find answers to a list of questions within a provided block of text.


Data Source (Custom Dataset): Use the following Python variables as your data.


Python


context = ''' The Transformer architecture was introduced in the 2017 paper "Attention Is All You Need". It is based on a self-attention mechanism, which replaces the recurrent layers (RNNs) used in previous models. This design allows for significantly more parallelization during training. The original model had an encoder-decoder structure, but later models like BERT used only the encoder, and models like GPT used only the decoder. '''

questions = ["When was transformer architecture introduced?", "What is it based on?"]
Expectations: Your submitted script should:

· Import AutoTokenizer and AutoModelForQuestionAnswering.

· Load the distilbert-base-cased-distilled-squad model and its corresponding tokenizer.

· Loop through each question in the questions list.

· Inside the loop, tokenize the question and context pair together.

· Pass the tokenized inputs to the model to get the start_logits and end_logits output.

· Use the tokenizer to decode the token span (from the start token to the end token) back into a human-readable answer string.

· Print the question and its corresponding answer string.
"""




from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch


context = """ The Transformer architecture was introduced in the 2017 paper "Attention Is All You Need". It is based on a self-attention mechanism, which replaces the recurrent layers (RNNs) used in previous models. This design allows for significantly more parallelization during training. The original model had an encoder-decoder structure, but later models like BERT used only the encoder, and models like GPT used only the decoder. """

questions = ["When was transformer architecture introduced?", "What is it based on?"]

# Load the pretrained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased-distilled-squad")
model = AutoModelForQuestionAnswering.from_pretrained("distilbert-base-cased-distilled-squad")

# Loop through each question and find the answer
for question in questions:
    # Tokenize the question and context
    inputs = tokenizer(question, context, return_tensors="pt")

    # Get the model's predictions
    outputs = model(**inputs)
    answer_start_scores = outputs.start_logits
    answer_end_scores = outputs.end_logits

    # Get the most likely answer span
    answer_start = torch.argmax(answer_start_scores)
    answer_end = torch.argmax(answer_end_scores) + 1  # Added 1 to include the end token

    # Convert Tokens Back to Human Text
    answer = tokenizer.decode(inputs["input_ids"][0, answer_start:answer_end])

    # Print the question and answer
    print(f"Question: {question}")
    print(f"Answer: {answer}\n")