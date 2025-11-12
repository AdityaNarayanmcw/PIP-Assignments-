Task 1: Inference with a Pretrained Model

Google Colab Link:
https://colab.research.google.com/drive/1Nm5CVqpyP8v3T3U3vpkg-O8IkpaIvsW3#scrollTo=O9qWx6LvQ4m6

This script performs Extractive Question Answering using the distilbert-base-cased-distilled-squad model.

Script Highlights:

  * Imports AutoTokenizer and AutoModelForQuestionAnswering from transformers
  
  * Loads the pretrained model and its tokenizer
  
  * Iterates through a list of questions
  
  * Tokenizes each (question, context) pair
  
  * Runs inference to obtain start_logits and end_logits
  
  * Extracts the most likely answer span from the model output
  
  * Decodes the answer into readable text
  
  * Prints each question with its corresponding answer


Task 2: Fine-Tuning a Transformer Model (Multiclass Emotion Classification) Task Description: Fine-tune the distilbert-base-uncased model for multiclass text classification. Your task is to train a model to predict one of six emotions (sadness, joy, love, anger, fear, surprise) based on the text. This task requires adapting the standard binary classification workflow to a multiclass problem.


Data Source:
Dataset: emotion (also known as dair-ai/emotion)

Source: Load directly from the Hugging Face datasets library using load_dataset("emotion").

Expectations: Your submitted script should be a complete, runnable training file that:
· Loads the emotion dataset.

· Loads the AutoTokenizer for distilbert-base-uncased.

· Loads AutoModelForSequenceClassification from distilbert-base-uncased.

· Defines a preprocessing function that tokenizes the text column. (Note: The label column in this dataset is already in the correct integer format, so no mapping is needed).

· Applies this preprocessing function to the dataset using .map().

· Defines a DataCollatorWithPadding.

· Defines TrainingArguments to configure the training process.

· Instantiates the Trainer with the model, arguments, datasets, tokenizer, and data collator.

· Calls trainer.train() to begin the fine-tuning process.

Dataset: emotion (also known as dair-ai/emotion)
Source: Load directly from the Hugging Face datasets library using load_dataset("emotion").
· Loads the emotion dataset.
source:
https://huggingface.co/docs/hub/datasets-usage
https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/text_classification.ipynb#scrollTo=UmvbnJ9JIrJd
"""



