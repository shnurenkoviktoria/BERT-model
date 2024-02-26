# Text Classification with BERT

This README demonstrates text classification using BERT (Bidirectional Encoder Representations from Transformers).

## Setup

- Keras and TensorFlow libraries are imported.
- The availability of GPU devices is checked.

## Data Preparation

- The IMDb dataset is loaded using the Keras IMDb dataset loader.
- The integer sequences representing words are converted to text.
- Text sequences are tokenized using the BERT tokenizer.
- Tokenized sequences are padded and truncated to a maximum length of 512 tokens.

## BERT Model

- The pre-trained BERT model (`bert-base-uncased`) is loaded using the `TFBertModel` class from the Transformers library.
- The BERT layers are frozen to prevent fine-tuning during training.
- A dense layer with a sigmoid activation function is added to the BERT model for binary classification.

## Training

- The BERT-based classifier model is compiled with the Adam optimizer and binary cross-entropy loss function.
- The model is trained on the tokenized training data for one epoch with a batch size of 32 and a validation split of 20%.

## Evaluation

- The trained model is evaluated on the tokenized test data.
- Test accuracy is computed and printed.

## Dependencies

- TensorFlow
- Keras
- Hugging Face Transformers
