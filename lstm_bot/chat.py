"""
Text Based Chat Interface to LSTM

"""
from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
import numpy as np
import random
import sys
import pickle

MAXLEN = 40
CHARS = 57
DIVERSITY = 0.2
index_list = pickle.load(open('index_list.pkl', 'rb'))
char_indices = index_list[0]
indices_char = index_list[1]

model = Sequential()
model.add(LSTM(512, return_sequences=True, input_shape=(MAXLEN, CHARS)))
model.add(Dropout(0.2))
model.add(LSTM(512, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(CHARS))
model.add(Activation('softmax'))

model.load_weights("model_weights.h5")
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

def sample(a, temperature=1.0):
    # helper function to sample an index from a probability array
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    return np.argmax(np.random.multinomial(1, a, 1))

def lstm_predict(user_input):

    # pad user input to put text at the end of 40 chars
    input_len = len(user_input)
    pad_len = MAXLEN - input_len
    pad = " "*pad_len
    question = pad + user_input

    answer = ""
    answer += question

    for i in range(400):
            x = np.zeros((1, MAXLEN, CHARS))
            for t, char in enumerate(question):
                x[0, t, char_indices[char]] = 1.

            preds = model.predict(x, verbose=0)[0]
            next_index = sample(preds, DIVERSITY)
            next_char = indices_char[next_index]

            answer += next_char
            question = question[1:] + next_char

    return answer




def chat_loop():
    """

    :return:
    """
    print("God is dead, so am I...or am I?  Ask me a question!")

    while True:
        user_input = str(input(">")).lower().strip()

        if user_input in ['exit', 'quit', 'stop']:
            break

        output = lstm_predict(user_input)
        print(output)
        # print("There is always some madness in love. But there is also always some reason in madness.")






if __name__ == "__main__":
    chat_loop()