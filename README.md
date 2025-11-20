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

google colab link : https://colab.research.google.com/drive/1oiIpG99zNsqi9zz1NxhO7ZmW_-RLo26g#scrollTo=1d82b3b0

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

Task 3: Retrieval-Augmented Generation (RAG) Script

Open ai google colab link: https://colab.research.google.com/drive/1vh7JzCtyJKQ1rS4KfArCvYPoC1Oz0T3D#scrollTo=WxJ9eelhQhVj

huggingface google colab link : https://colab.research.google.com/drive/1vaeBtuPokMFFdOWo3AgBE8_Llwv_0YX6#scrollTo=V5mRSGWur1w5

Task Description: Write a single Python script that implements a complete, in-memory RAG pipeline. The script must answer questions based only on the content of a specific online document. This task has two phases.


Data Source:

Document URL: https://lilianweng.github.io/posts/2023-06-23-agent/


Expectations: Your script must perform the full "Load, Split, Embed, Store, Retrieve, Generate" workflow.


Core Pipeline (must be implemented):


· Load: Use a DocumentLoader (e.g., WebBaseLoader) to load the text content from the URL.

· Split: Use a TextSplitter (e.g., RecursiveCharacterTextSplitter) to break the document into smaller chunks.

· Embed & Store: Use an embedding model and an in-memory vector store (e.g., FAISS or Chroma) to create a searchable index of the chunks.

· Retrieve & Generate: Create a RetrievalQA chain (or equivalent) that connects a retriever (from your vector store) and an LLM (model of your choice) to answer questions.

Phase 1 (For your verification):

· Hardcode a question in your script, such as "What are the three main components of an LLM agent?"

· Run the script and verify that your RAG chain prints a correct, grounded answer.


Phase 2 (Final submission):

· Remove the hardcoded question.

· Modify the script so it accepts a question as user input (e.g., using Python's built-in input() function).

· The script should then take this user input, run the RAG chain, and print the final answer to the console.

· Please submit your script in this final "Phase 2" state.

RAG Evaluation using RAGAS
Evaluation Overview:
I evaluated the pipeline across five key metrics to measure both retrieval quality and generation accuracy:
Context Recall - Measures how well the retrieval captures all relevant information
Faithfulness - Assesses whether generated answers are grounded in the retrieved context
Factual Correctness - Evaluates the accuracy of facts in the generated responses
Answer Relevancy - Determines how relevant the answers are to the user queries
Context Precision - Measures the precision of retrieved contexts



