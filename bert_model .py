import keras
import tensorflow as tf
from keras.layers import Input, Dense
from keras.models import Model
from transformers import BertTokenizer, TFBertModel


print(tf.config.list_physical_devices("GPU"))

(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data()


x_train = [" ".join([str(word) for word in sequence]) for sequence in x_train]
x_test = [" ".join([str(word) for word in sequence]) for sequence in x_test]


tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

x_train_tokens = tokenizer(
    x_train, padding=True, truncation=True, return_tensors="tf", max_length=512
)
x_test_tokens = tokenizer(
    x_test, padding=True, truncation=True, return_tensors="tf", max_length=512
)


bert_model = TFBertModel.from_pretrained("bert-base-uncased")


for layer in bert_model.layers:
    layer.trainable = False

inputs = Input(shape=(512,), dtype=tf.int32)
bert_output = bert_model(inputs)["last_hidden_state"]
pooled_output = bert_output[:, 0, :]
outputs = Dense(1, activation="sigmoid")(pooled_output)


bert_classifier_model = Model(inputs=inputs, outputs=outputs)


bert_classifier_model.compile(
    optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
)

bert_classifier_model.fit(
    x_train_tokens["input_ids"], y_train, epochs=1, batch_size=32, validation_split=0.2
)


accuracy = bert_classifier_model.evaluate(x_test_tokens["input_ids"], y_test)[1]
print(f"Test Accuracy: {accuracy}")
