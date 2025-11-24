# Masters Thesis – Recognizing emotions in text using classical and large language models

This repository contains the full codebase used in my master’s thesis, which compares traditional neural architectures (Feed-Forward, CNN, LSTM) with a fine-tuned transformer-based large language model for multiclass emotion recognition in text.

The project includes:

* data preprocessing and cleaning
* class balancing through undersampling
* implementation and training of multiple deep learning models
* evaluation and comparison of classification performance
* reproduction of all results presented in the thesis

---

## Dataset & Preprocessing

* The dataset consists of short text samples labeled with emotional categories.
* Duplicate entries were removed.
* To ensure fair comparison between models, **all emotion classes were balanced** using undersampling, resulting in equal representation of each class.
* Texts were tokenized using:

  * **TensorFlow tokenizer** for traditional neural networks (Feed-Forward, CNN, LSTM)
  * **Hugging Face tokenizer** for transformer fine-tuning

---

## Models Evaluated

The project implements and compares the following architectures:

### 1. Feed-Forward Neural Network

Basic baseline model using learned embeddings.

### 2. Convolutional Neural Network (CNN)

Single-branch 1D convolution for text feature extraction.

### 3. Deep CNN

Extended CNN architecture with additional dense layers.

### 4. LSTM

Standard unidirectional Long Short-Term Memory network.

### 5. Deep LSTM

Stacked LSTM layers with additional dense blocks.

### 6. Fine-tuned Transformer

DistilBERT (`distilbert-base-uncased`) fine-tuned for multi-class emotion recognition.

---

## Performance (Accuracy)

All models were evaluated on the same balanced test set.

| Model                  | Accuracy   |
| ---------------------- | ---------- |
| Feed-Forward Network   | **78.53%** |
| CNN                    | **97.51%** |
| Deep CNN               | **96.34%** |
| LSTM                   | **98.44%** |
| Deep LSTM              | **98.68%** |
| Fine-tuned Transformer | **99.33%** |

---

