import numpy as np
from tensorflow import keras
from tensorflow.keras import layers


class AbstractMachine:
    """Artificial intelligence! to generate abstracts for imaginary papers. Takes a random sample of abstracts
    (which we can easily query from PubMed) then fit a recurrent neural network (RNN). The RNN can then generate
    abstracts.

    Here, I am cheating by adding the special character `@` to determine the 'end' of the abstract. It is a lazy way
    to determine the 'end' for both the machine and when we pull out the abstract.

    """
    def __init__(self, list_of_abstracts, length=40, step_size=3):
        self.abstracts = [abstract.lower() for abstract in list_of_abstracts]
        self.max_length = length
        self.step_size = step_size

        # Flagging all unique characters
        self.text = " ".join(self.abstracts)
        self.chars = sorted(list(set(self.text)))
        self.char_index = dict((c, i) for i, c in enumerate(self.chars))
        self.index_char = dict((i, c) for i, c in enumerate(self.chars))

        # Extracting chunks of text
        sentences, next_char = [], []
        for abs_i in self.abstracts:
            for i in range(0, len(abs_i) - length, step_size):
                sentences.append(abs_i[i: i+length])
                next_char.append(abs_i[i+length])

        # Creating corresponding binary variables
        self.x = np.zeros((len(sentences), length, len(self.chars)), dtype=np.bool)
        self.y = np.zeros((len(sentences), len(self.chars)), dtype=np.bool)
        for i, sentence in enumerate(sentences):
            for t, char in enumerate(sentence):
                self.x[i, t, self.char_index[char]] = 1
            self.y[i, self.char_index[next_char[i]]] = 1

        self.fit_model = None

    def fit(self, epochs=20, batch_size=128):
        model = keras.Sequential(
            [keras.Input(shape=(self.max_length, len(self.chars))),
             layers.LSTM(batch_size),
             layers.Dense(len(self.chars), activation="softmax"), ])

        # Setting learning rate to decay over epochs
        lr_schedule = keras.optimizers.schedules.ExponentialDecay(0.005, decay_steps=1000,
                                                                  decay_rate=0.95,
                                                                  staircase=True)

        optimizer = keras.optimizers.RMSprop(learning_rate=lr_schedule,
                                             clipnorm=100)
        model.compile(loss="categorical_crossentropy", optimizer=optimizer)

        model.fit(self.x, self.y, batch_size=batch_size, epochs=epochs)
        self.fit_model = model

    def predict_text(self, start_text, creativity, max_characters):
        start_text = start_text.lower()
        if len(start_text) < self.max_length:
            sentence = start_text
            # raise ValueError("Input text is shorter than the required starting text...")
        elif len(start_text) > self.max_length:
            sentence = start_text[-40:]
        else:
            sentence = start_text
        generated = "" + start_text

        for i in range(max_characters):
            x_pred = np.zeros((1, self.max_length, len(self.chars)))
            for t, char in enumerate(sentence):
                x_pred[0, t, self.char_index[char]] = 1.

            preds = self.fit_model.predict(x_pred, verbose=0)[0]
            next_index = self.sample(preds, creativity)
            next_char = self.index_char[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char

        return generated.capitalize()

    @staticmethod
    def sample(preds, temperature=1.0):
        preds = np.asarray(preds).astype("float64")
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)

