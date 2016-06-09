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

hamilton_txt = open('Hamilton_libretto.txt').read().lower()
les_mis_txt = open('Les_Mis_libretto.txt').read().lower()
combined_txt = hamilton_txt + les_mis_txt
print('corpus length:', len(combined_txt))

chars = set(combined_txt)
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))
maxlen = 80
DIVERSITY = 1

print(char_indices['c'])
print(indices_char[5])
# reload les mis model
print('Build model...')
lm_model = Sequential()
lm_model.add(LSTM(512, return_sequences=True, input_shape=(maxlen, len(chars))))
lm_model.add(Dropout(0.5))
lm_model.add(LSTM(512, return_sequences=True))
lm_model.add(Dropout(0.5))
lm_model.add(LSTM(512, return_sequences=False))
lm_model.add(Dropout(0.5))
lm_model.add(Dense(len(chars)))
lm_model.add(Activation('softmax'))
lm_model.load_weights("les_mis_model_weights3l_long.h5")
lm_model.compile(loss='categorical_crossentropy', optimizer='rmsprop')


# reload hamilton model
print('Build model...')
ham_model = Sequential()
ham_model.add(LSTM(512, return_sequences=True, input_shape=(maxlen, len(chars))))
ham_model.add(Dropout(0.5))
ham_model.add(LSTM(512, return_sequences=True))
ham_model.add(Dropout(0.5))
ham_model.add(LSTM(512, return_sequences=False))
ham_model.add(Dropout(0.5))
ham_model.add(Dense(len(chars)))
ham_model.add(Activation('softmax'))
ham_model.load_weights("hamilton_model_weights3l_long.h5")
ham_model.compile(loss='categorical_crossentropy', optimizer='rmsprop')



def sample(a, temperature=1.0):
    # helper function to sample an index from a probability array
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    return np.argmax(np.random.multinomial(1, a, 1))


def lstm_predict(input, model):
    """
        input:  What we can use to seed the LSTM
        model:  The model we are generating text from

    """

    # generate lstm output
    input_len = len(input)
    pad_len = maxlen - input_len
    pad = " "*pad_len
    question = pad + input

    answer = ""
    answer += question

    for i in range(400):
            x = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(question):
                x[0, t, char_indices[char]] = 1.

            preds = model.predict(x, verbose=0)[0]
            next_index = sample(preds, DIVERSITY)
            next_char = indices_char[next_index]

            answer += next_char
            question = question[1:] + next_char

    return answer




def rap_battle():
    """

    :return:
    """
    input = "pardon me"
    output = lstm_predict(input, ham_model)
    print(output)






if __name__ == "__main__":
    rap_battle()