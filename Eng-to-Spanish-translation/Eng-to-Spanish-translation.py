#!/usr/bin/env python
# coding: utf-8

# # Built a seq2seq model for Language translation.
# 
# Task: Change LSTM model to Bidirectional LSTM Model and Translate English to Spanish

# ## 1. Data preparation 

from google.colab import drive
drive.mount('/content/drive')


# ### 1.1. Load and clean text

import re
import string
from unicodedata import normalize
import numpy

# load doc into memory
def load_doc(filename):
    # open the file as read only
    file = open(filename, mode='rt', encoding='utf-8')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text


# split a loaded document into sentences
def to_pairs(doc):
    lines = doc.strip().split('\n')
    pairs = [line.split('\t') for line in  lines]
    return pairs

def clean_data(lines):
    cleaned = list()
    # prepare regex for char filtering
    re_print = re.compile('[^%s]' % re.escape(string.printable))
    # prepare translation table for removing punctuation
    table = str.maketrans('', '', string.punctuation)
    for pair in lines:
        clean_pair = list()
        for line in pair:
            # normalize unicode characters
            line = normalize('NFD', line).encode('ascii', 'ignore')
            line = line.decode('UTF-8')
            # tokenize on white space
            line = line.split()
            # convert to lowercase
            line = [word.lower() for word in line]
            # remove punctuation from each token
            line = [word.translate(table) for word in line]
            # remove non-printable chars form each token
            line = [re_print.sub('', w) for w in line]
            # remove tokens with numbers in them
            line = [word for word in line if word.isalpha()]
            # store as string
            clean_pair.append(' '.join(line))
        cleaned.append(clean_pair)
    return numpy.array(cleaned)



# e.g., filename = 'Data/deu.txt'
filename = '/content/drive/MyDrive/Data/spa.txt'

# e.g., n_train = 20000
n_train = 20000

# load dataset
doc = load_doc(filename)
# split into Language1-Language2 pairs
pairs = to_pairs(doc)
# clean sentences
clean_pairs = clean_data(pairs)[0:n_train, :]

for i in range(3000, 3010):
    print('[' + clean_pairs[i, 0] + '] => [' + clean_pairs[i, 1] + ']')


input_texts = clean_pairs[:, 0]
target_texts = ['\t' + text + '\n' for text in clean_pairs[:, 1]]
# print(target_texts)
# print(input_texts)
print('Length of input_texts:  ' + str(input_texts.shape))
print('Length of target_texts: ' + str(input_texts.shape))

max_encoder_seq_length = max(len(line) for line in input_texts)
max_decoder_seq_length = max(len(line) for line in target_texts)

print('max length of input  sentences: %d' % (max_encoder_seq_length))
print('max length of target sentences: %d' % (max_decoder_seq_length))


# **Remark:**  Two lists of sentences: input_texts and target_texts

# 2. Text processing
# 2.1. Convert texts to sequences
# 
# - Input: A list of $n$ sentences (with max length $t$).
# - It is represented by a $n\times t$ matrix after the tokenization and zero-padding.

from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# encode and pad sequences
def text2sequences(max_len, lines):
    tokenizer = Tokenizer(char_level=True, filters='')
    tokenizer.fit_on_texts(lines)
    seqs = tokenizer.texts_to_sequences(lines)
    seqs_pad = pad_sequences(seqs, maxlen=max_len, padding='post')
    return seqs_pad, tokenizer.word_index


encoder_input_seq, input_token_index = text2sequences(max_encoder_seq_length, 
                                                      input_texts)
decoder_input_seq, target_token_index = text2sequences(max_decoder_seq_length, 
                                                       target_texts)
# print(encoder_input_seq[1])
# print(target_token_index)
# print(input_token_index)
# print(decoder_input_seq[1])
print('shape of encoder_input_seq: ' + str(encoder_input_seq.shape))
print('shape of input_token_index: ' + str(len(input_token_index)))
print('shape of decoder_input_seq: ' + str(decoder_input_seq.shape))
print('shape of target_token_index: ' + str(len(target_token_index)))

num_encoder_tokens = len(input_token_index) + 1
num_decoder_tokens = len(target_token_index) + 1

print('num_encoder_tokens: ' + str(num_encoder_tokens))
print('num_decoder_tokens: ' + str(num_decoder_tokens))


# **Remark:** To this end, the input language and target language texts are converted to 2 matrices. 
# 
# - Their number of rows are both n_train.
# - Their number of columns are respective max_encoder_seq_length and max_decoder_seq_length.

target_texts[100]


decoder_input_seq[100, :]


#  2.2. One-hot encode
# 
# - Input: A list of $n$ sentences (with max length $t$).
# - It is represented by a $n\times t$ matrix after the tokenization and zero-padding.
# - It is represented by a $n\times t \times v$ tensor ($t$ is the number of unique chars) after the one-hot encoding.

from keras.utils import to_categorical

# one hot encode target sequence
def onehot_encode(sequences, max_len, vocab_size):
    n = len(sequences)
    data = numpy.zeros((n, max_len, vocab_size))
    for i in range(n):
        data[i, :, :] = to_categorical(sequences[i], num_classes=vocab_size)
    return data

encoder_input_data = onehot_encode(encoder_input_seq, max_encoder_seq_length, num_encoder_tokens)
decoder_input_data = onehot_encode(decoder_input_seq, max_decoder_seq_length, num_decoder_tokens)

decoder_target_seq = numpy.zeros(decoder_input_seq.shape)
decoder_target_seq[:, 0:-1] = decoder_input_seq[:, 1:]
decoder_target_data = onehot_encode(decoder_target_seq, max_decoder_seq_length, num_decoder_tokens)

print(encoder_input_data.shape)
print(decoder_input_data.shape)


#  3. Build the networks (for training)

#  3.1. Encoder network
# 
# - Input:  one-hot encode of the input language
# 
# - Return: 
# 
#     -- output (all the hidden states   $h_1, \cdots , h_t$) are always discarded
#     
#     -- the final hidden state  $h_t$
#     
#     -- the final conveyor belt $c_t$

# **Trained model using Bidirectional LSTM model**

from keras.layers import Input, LSTM, Bidirectional, Concatenate
from keras.models import Model

latent_dim = 256

# inputs of the encoder network
encoder_inputs = Input(shape=(None, num_encoder_tokens), name='encoder_inputs')
# set the LSTM layer
encoder_bilstm = Bidirectional(LSTM(latent_dim, return_state=True,dropout=0.5, name='encoder_bidrectional_lstm'))
_, forward_h, forward_c, backward_h, backward_c = encoder_bilstm(encoder_inputs)

state_h = Concatenate()([forward_h, backward_h])
state_c = Concatenate()([forward_c, backward_c])

# build the encoder network model
encoder_model = Model(inputs=encoder_inputs,outputs=[state_h, state_c], name='encoder')


# Print a summary and save the encoder network structure to "./encoder.pdf"

from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot, plot_model

SVG(model_to_dot(encoder_model, show_shapes=False).create(prog='dot', format='svg'))

plot_model(
    model=encoder_model, show_shapes=False,
    to_file='encoder.pdf'
)

encoder_model.summary()


#  3.2. Decoder network
# 
# - Inputs:  
# 
#     -- one-hot encode of the target language
#     
#     -- The initial hidden state $h_t$ 
#     
#     -- The initial conveyor belt $c_t$ 
# 
# - Return: 
# 
#     -- output (all the hidden states) $h_1, \cdots , h_t$
# 
#     -- the final hidden state  $h_t$ (discarded in the training and used in the prediction)
#     
#     -- the final conveyor belt $c_t$ (discarded in the training and used in the prediction)


from keras.layers import Input, LSTM, Dense
from keras.models import Model

# inputs of the decoder network
decoder_input_h = Input(shape=(latent_dim *2,), name='decoder_input_h')
decoder_input_c = Input(shape=(latent_dim *2,), name='decoder_input_c')
decoder_input_x = Input(shape=(None, num_decoder_tokens), name='decoder_input_x')

# set the LSTM layer
decoder_lstm = LSTM(latent_dim *2, return_sequences=True, return_state=True, dropout=0.5, name='decoder_lstm')
decoder_lstm_outputs, state_h, state_c = decoder_lstm(decoder_input_x, initial_state=[decoder_input_h, decoder_input_c])

# set the dense layer
decoder_dense = Dense(num_decoder_tokens, activation='softmax', name='decoder_dense')
decoder_outputs = decoder_dense(decoder_lstm_outputs)

# build the decoder network model
decoder_model = Model(inputs=[decoder_input_x, decoder_input_h, decoder_input_c],
                      outputs=[decoder_outputs, state_h, state_c],
                      name='decoder')


# Print a summary and save the encoder network structure to "./decoder.pdf"


from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot, plot_model

SVG(model_to_dot(decoder_model, show_shapes=False).create(prog='dot', format='svg'))

plot_model(
    model=decoder_model, show_shapes=False,
    to_file='decoder.pdf'
)

decoder_model.summary()


#  3.3. Connect the encoder and decoder

# input layers
encoder_input_x = Input(shape=(None, num_encoder_tokens), name='encoder_input_x')
decoder_input_x = Input(shape=(None, num_decoder_tokens), name='decoder_input_x')

# connect encoder to decoder
encoder_final_states = encoder_model([encoder_input_x])
decoder_lstm_output, _, _ = decoder_lstm(decoder_input_x, initial_state=encoder_final_states)
decoder_pred = decoder_dense(decoder_lstm_output)

model = Model(inputs=[encoder_input_x, decoder_input_x], 
              outputs=decoder_pred, 
              name='model_training')

from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot, plot_model

SVG(model_to_dot(model, show_shapes=False).create(prog='dot', format='svg'))

plot_model(
    model=model, show_shapes=False,
    to_file='model_training.pdf'
)

model.summary()


# ### 3.4. Fit the model on the bilingual dataset
# 
# - encoder_input_data: one-hot encode of the input language
# 
# - decoder_input_data: one-hot encode of the input language
# 
# - decoder_target_data: labels (left shift of decoder_input_data)
# 
# - tune the hyper-parameters
# 
# - stop when the validation loss stop decreasing.

print('shape of encoder_input_data' + str(encoder_input_data.shape))
print('shape of decoder_input_data' + str(decoder_input_data.shape))
print('shape of decoder_target_data' + str(decoder_target_data.shape))


# ## 4. Make predictions
# 
# - In this section, you need to complete section 4.2 to translate English to the target language.
# 
# 
# ### 4.1. Translate English to XXX
# 
# 1. Encoder read a sentence (source language) and output its final states, $h_t$ and $c_t$.
# 2. Take the [star] sign "\t" and the final state $h_t$ and $c_t$ as input and run the decoder.
# 3. Get the new states and predicted probability distribution.
# 4. sample a char from the predicted probability distribution
# 5. take the sampled char and the new states as input and repeat the process (stop if reach the [stop] sign "\n").

model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

model.fit([encoder_input_data, decoder_input_data],  # training data
          decoder_target_data,                       # labels (left shift of the target sequences)
          batch_size=64, epochs=50, validation_split=0.2)

model.save('seq2seq.h5')

# Reverse-lookup token index to decode sequences back to something readable.
reverse_input_char_index = dict((i, char) for char, i in input_token_index.items())
reverse_target_char_index = dict((i, char) for char, i in target_token_index.items())


# **Implemented Multinomial Sampling on the dataset decoded sequence**

def softmax(z):
   return numpy.exp(z)/sum(numpy.exp(z))
def decode_sequence(input_seq):
    states_value = encoder_model.predict(input_seq)

    target_seq = numpy.zeros((1, 1, num_decoder_tokens))
    target_seq[0, 0, target_token_index['\t']] = 1.
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        # this line of code is greedy selection
        # try to use multinomial sampling instead (with temperature)
        # sampled_token_index = numpy.argmax(output_tokens[0, -1, :])
        temperature = 0.5
        output_distribution = numpy.asarray(output_tokens[0, -1, :]).astype("float64")
        output_distribution =  numpy.log(output_distribution)/temperature
        reweighted_conditional_probability = softmax(output_distribution)
        sampled_token_index = numpy.argmax(numpy.random.multinomial(1, reweighted_conditional_probability, 1))
        if(sampled_token_index == 0):
            reweighted_conditional_probability[sampled_token_index] = 0
            second_highest_prob_index = numpy.argmax(reweighted_conditional_probability)
            sampled_char = reverse_target_char_index[second_highest_prob_index]
        else:
            sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char

        if (sampled_char == '\n' or
           len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True

        target_seq = numpy.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

        states_value = [h, c]

    return decoded_sentence


for seq_index in range(2100, 2120):
    # Take one sequence (part of the training set)
    # for trying out decoding.
    input_seq = encoder_input_data[seq_index: seq_index + 1]
    decoded_sentence = decode_sequence(input_seq)
    print('-')
    print('English:       ', input_texts[seq_index])
    print('Spanish (true): ', target_texts[seq_index][1:-1])
    print('Spanish (pred): ', decoded_sentence[0:-1])


# ### 4.2. Translate an English sentence to the target language 
# 
# 1. Tokenization
# 2. One-hot encode
# 3. Translate

from keras.preprocessing.text import Tokenizer
input_sentence = 'I love you'
input_ = input_sentence
input_ = input_.lower() #lower case 

def word_to_index(max_len, line):   #tokenization using above trainable word_index dict.
  tokenizer = Tokenizer(char_level=True, filters='')
  tokenizer.fit_on_texts(line)
  input_dict = {}
  input_dict = dict((char, index) for char, index in input_token_index.items() for i in line[0] if char == i) #to initialize values to particular sentence
  seqs = tokenizer.texts_to_sequences(line) 
  seqs_pad = pad_sequences(seqs, maxlen=max_len, padding='post')
  return seqs_pad, input_dict

input_sequence, sequence_dict = word_to_index(len(input_sentence), [input_sentence]) 
input_x = onehot_encode(input_sequence, len(input_sentence), num_encoder_tokens)
translated_sentence = decode_sequence(input_x) 

print('source sentence is: ' + input_sentence)
print('translated sentence is: ' + translated_sentence)


# # 5. Evaluate the translation using BLEU score
# 
# Reference:
# 
# https://machinelearningmastery.com/calculate-bleu-score-for-text-python/
# 
# https://en.wikipedia.org/wiki/BLEU

# ### 5.1. Partition the dataset to training, validation, and test. Build new token index.

n_data = 60000
# load dataset
doc = load_doc(filename)

# split into Language1-Language2 pairs
pairs = to_pairs(doc)

# clean sentences
clean_pairs = clean_data(pairs)[0:n_data, :]
for i in range(4000, 4010):
    print('[' + clean_pairs[i, 0] + '] => [' + clean_pairs[i, 1] + ']')

input_texts = clean_pairs[:, 0]
target_texts = ['\t' + text + '\n' for text in clean_pairs[:, 1]]

# Training data of aroud 40000 records
print('Length of input_texts:  ' + str(input_texts.shape))
print('Length of target_texts: ' + str(input_texts.shape))


# **Randomly distributed dataset.**

rand_indices = numpy.random.permutation(60000)
train_input_texts = input_texts[0:50000]
train_target_texts = target_texts[0:50000]
test_input_texts = input_texts[50000:60000]
test_target_text = target_texts[50000:60000]

#Executing to get max encoder and decoder length
max_encoder_seq_length_train = max(len(line) for line in train_input_texts)
max_decoder_seq_length_train = max(len(line) for line in train_target_texts)
print('max length of input  sentences: %d' % (max_encoder_seq_length_train))
print('max length of target sentences: %d' % (max_decoder_seq_length_train))

#decode train dataset sequences
encoder_input_seqs, input_token_indexs = text2sequences(max_encoder_seq_length_train, train_input_texts)
decoder_input_seqs, target_token_indexs = text2sequences(max_decoder_seq_length_train,train_target_texts)

print('shape of encoder_input_seq: ' + str(encoder_input_seqs.shape))
print('shape of input_token_index: ' + str(len(input_token_indexs)))
print('shape of decoder_input_seq: ' + str(decoder_input_seqs.shape))
print('shape of target_token_index: ' + str(len(target_token_indexs)))

num_encoder_token = len(input_token_indexs) + 1
num_decoder_token = len(target_token_indexs) + 1

print('num_encoder_tokens: ' + str(num_encoder_token))
print('num_decoder_tokens: ' + str(num_decoder_token))
print('Target value:'+ str(train_target_texts[4000]))
print(decoder_input_seqs[4000, :])

#decode the train data
encoder_input_data_train = onehot_encode(encoder_input_seqs, max_encoder_seq_length_train, num_encoder_token)
decoder_input_data_train = onehot_encode(decoder_input_seqs, max_decoder_seq_length_train, num_decoder_token)

decoder_target_seqs = numpy.zeros(decoder_input_seqs.shape)
decoder_target_seqs[:, 0:-1] = decoder_input_seqs[:, 1:]
decoder_target_data_train = onehot_encode(decoder_target_seqs, max_decoder_seq_length_train, num_decoder_token)

print(encoder_input_data_train.shape)
print(decoder_input_data_train.shape)


# ### 5.2 Retrain your previous Bidirectional LSTM model with training and validation data and tune the parameters (learning rate, optimizer, etc) based on validation score.
# 
# 1. Use the model structure in section 3 to train a new model with new training and validation datasets.
# 2. Based on validation BLEU score or loss to tune parameters.

from keras.layers import Input, LSTM, Bidirectional, Concatenate
from keras.models import Model

latent_dim_train = 256
# inputs of the encoder network
encoder_train_inputs = Input(shape=(None, num_encoder_token), name='encoder_inputs_train')
# set the LSTM layer
encoder_train_bilstm = Bidirectional(LSTM(latent_dim_train, return_state=True,dropout=0.5, name='encoder_bidrectional_lstm_train'))
_, forward_h, forward_c, backward_h, backward_c = encoder_train_bilstm(encoder_train_inputs)

state_h_train = Concatenate()([forward_h, backward_h])
state_c_train = Concatenate()([forward_c, backward_c])

# build the encoder network model
encoder_model_train = Model(inputs=encoder_train_inputs,outputs=[state_h_train, state_c_train], name='encoder')

from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot, plot_model

SVG(model_to_dot(encoder_model_train, show_shapes=False).create(prog='dot', format='svg'))

plot_model(
    model=encoder_model_train, show_shapes=False,
    to_file='encoder.pdf'
)

encoder_model_train.summary()

from keras.layers import Input, LSTM, Dense
from keras.models import Model
from keras import optimizers
# inputs of the decoder network
decoder_input_h_train = Input(shape=(latent_dim_train *2,), name='decoder_input_h_train')
decoder_input_c_train = Input(shape=(latent_dim_train *2,), name='decoder_input_c_train')
decoder_input_x_train = Input(shape=(None, num_decoder_token), name='decoder_input_x_train')

# set the LSTM layer
decoder_lstm_train = LSTM(latent_dim_train *2, return_sequences=True, return_state=True, dropout=0.5, name='decoder_lstm_train')
decoder_lstm_outputs_train, state_h, state_c = decoder_lstm_train(decoder_input_x_train, initial_state=[decoder_input_h_train, decoder_input_c_train])

# set the dense layer
decoder_dense_train = Dense(num_decoder_token, activation='softmax', name='decoder_dense_train')
decoder_outputs_train = decoder_dense_train(decoder_lstm_outputs_train)

# build the decoder network model
decoder_model_train = Model(inputs=[decoder_input_x_train, decoder_input_h_train, decoder_input_c_train],
                      outputs=[decoder_outputs_train, state_h, state_c],
                      name='decoder')

from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot, plot_model

SVG(model_to_dot(decoder_model_train, show_shapes=False).create(prog='dot', format='svg'))

plot_model(
    model=decoder_model_train, show_shapes=False,
    to_file='decoder.pdf'
)

decoder_model_train.summary()

# input layers
encoder_input_x_train = Input(shape=(None, num_encoder_token), name='encoder_input_x_train')
decoder_input_x_train = Input(shape=(None, num_decoder_token), name='decoder_input_x_train')

# connect encoder to decoder
encoder_final_states = encoder_model_train([encoder_input_x_train])
decoder_lstm_output_train, _, _ = decoder_lstm_train(decoder_input_x_train, initial_state=encoder_final_states)
decoder_pred = decoder_dense_train(decoder_lstm_output_train)

model_train = Model(inputs=[encoder_input_x_train, decoder_input_x_train], 
              outputs=decoder_pred, 
              name='model_training')


from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot, plot_model

SVG(model_to_dot(model_train, show_shapes=False).create(prog='dot', format='svg'))

plot_model(
    model=model_train, show_shapes=False,
    to_file='model_training.pdf'
)

model_train.summary()

# rand_indices = numpy.random.permutation(50000)
# train_indices = rand_indices[0:40000]
# valid_indices = rand_indices[40000:50000]


x_input_train_encoder = encoder_input_data_train[0:40000, :, :]
x_input_train_decoder = decoder_input_data_train[0:40000, :, :]

x_input_val_encoder = encoder_input_data_train[40000:50000, :, :]
x_input_val_decoder = decoder_input_data_train[40000:50000, :, :]

x_target_train_decoder = decoder_target_data_train[0:40000, :, :]
x_target_val_decoder = decoder_target_data_train[40000:50000, :, :]

print('shape of training  encoder dataset', str(x_input_train_encoder.shape))
print('shape of training decoder dataset', str(x_input_train_decoder.shape))

print('shape of validation encoder dataset', str(x_input_val_encoder.shape))
print('shape of validation decoder dataset', str(x_input_val_decoder.shape))

model_train.compile(optimizer=optimizers.Adam(learning_rate=0.003),loss='categorical_crossentropy')

model_train.fit([x_input_train_encoder, x_input_train_decoder],  # training data
          x_target_train_decoder,                       # labels (left shift of the target sequences)
          batch_size=125, epochs=50, validation_data=([x_input_val_encoder, x_input_val_decoder],x_target_val_decoder), validation_steps=20)

model_train.save('seq2seq.h5')


# Reverse-lookup token index to decode sequences back to something readable.
reverse_input_char_index = dict((i, char) for char, i in input_token_indexs.items())
reverse_target_char_index = dict((i, char) for char, i in target_token_indexs.items())


# **Multinomial sampling for decoding the sequences.**


def softmax(z):
   return numpy.exp(z)/sum(numpy.exp(z))

def decode_sequence(input_seq):
    states_value = encoder_model_train.predict(input_seq)

    target_seq = numpy.zeros((1, 1, num_decoder_tokens))
    target_seq[0, 0, target_token_index['\t']] = 1.
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model_train.predict([target_seq] + states_value)

        # this line of code is greedy selection
        # try to use multinomial sampling instead (with temperature)
        # sampled_token_index = numpy.argmax(output_tokens[0, -1, :])

        output_distribution = numpy.asarray(output_tokens[0, -1, :]).astype("float64")
        temperature = 0.5
        output_distribution = numpy.asarray(output_tokens[0, -1, :]).astype("float64")
        output_distribution =  numpy.log(output_distribution)/temperature
        reweighted_conditional_probability = softmax(output_distribution)
        sampled_token_index = numpy.argmax(numpy.random.multinomial(1, reweighted_conditional_probability, 1))
        if(sampled_token_index == 0):
          reweighted_conditional_probability[sampled_token_index] = 0
          second_highest_prob_index = numpy.argmax(reweighted_conditional_probability)
          sampled_char = reverse_target_char_index[second_highest_prob_index]
        else:
          sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char      
        # sampled_char = reverse_target_char_index[sampled_token_index]
        # decoded_sentence += sampled_char
        
        if (sampled_char == '\n' or len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True

        target_seq = numpy.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

        states_value = [h, c]
        
    return decoded_sentence

from nltk.translate.bleu_score import sentence_bleu
for seq_index in range(4000, 4010):
    # Take one sequence (part of the training set)
    # for trying out decoding
    input_seq = encoder_input_data_train[seq_index: seq_index + 1]
    decoded_sentence = decode_sequence(input_seq)
    print('-')
    print('English:       ', train_input_texts[seq_index])
    reference = train_target_texts[seq_index][1:-1]
    print('Spanish (true): ', reference)
    candidate = decoded_sentence[0:-1]
    print('Spanish (pred): ', candidate)


# ### 5.3 Evaluate the BLEU score using the test set.

# **Encode and decode the test dataset parameter using word2index dict of retrained model**

from keras.preprocessing.text import Tokenizer
# input_sentence = 'I love you'
# input_ = input_sentence
# input_ = input_.lower() #lower case 

def word2index_testencode(max_len, line):   #tokenization using above trainable word_index dict.
  tokenizer = Tokenizer(char_level=True, filters='')
  tokenizer.fit_on_texts(line)
  input_dict = {}
  input_dict = dict((char, index) for char, index in input_token_indexs.items() for i in line[0] if char == i) #to initialize values to particular sentence
  seqs = tokenizer.texts_to_sequences(line) 
  seqs_pad = pad_sequences(seqs, maxlen=max_len, padding='post')
  return seqs_pad, input_dict

def word2index_testdecode(max_len, line):   #tokenization using above trainable word_index dict.
  tokenizer = Tokenizer(char_level=True, filters='')
  tokenizer.fit_on_texts(line)
  input_dict = {}
  input_dict = dict((char, index) for char, index in target_token_indexs.items() for i in line[0] if char == i) #to initialize values to particular sentence
  seqs = tokenizer.texts_to_sequences(line) 
  seqs_pad = pad_sequences(seqs, maxlen=max_len, padding='post')
  return seqs_pad, input_dict
# input_x = onehot_encode(input_sequence, len(input_sentence), num_encoder_tokens)
# translated_sentence = decode_sequence(input_x) 

# print('source sentence is: ' + input_sentence)
# print('translated sentence is: ' + translated_sentence

# test_input_texts = input_texts[50000:60000]
# test_target_text = target_texts[50000:60000]
max_encoder_seq_length_test = max(len(line) for line in test_input_texts)
max_decoder_seq_length_test = max(len(line) for line in test_target_text)
print('max length of input  sentences: %d' % (max_encoder_seq_length_test))
print('max length of target sentences: %d' % (max_decoder_seq_length_test))

encoder_test_input_seqs, encode_sequence_dict = word2index_testencode(max_encoder_seq_length_test, test_input_texts) 
decoder_test_input_seqs, decode_sequence_dict = word2index_testdecode(max_decoder_seq_length_test,test_target_text)
x_test_encoder = onehot_encode(encoder_test_input_seqs, max_encoder_seq_length_test, num_encoder_token)
x_test_decoder = onehot_encode(decoder_test_input_seqs, max_decoder_seq_length_test, num_decoder_token)

decoder_target_test_seqs = numpy.zeros(decoder_test_input_seqs.shape)
decoder_target_test_seqs[:, 0:-1] = decoder_test_input_seqs[:, 1:]
x_target_test_decoder = onehot_encode(decoder_target_test_seqs, max_decoder_seq_length_test, num_decoder_token)


# **Loss encountered after testing the retraining model test datasetthe parameters**

loss = model_train.evaluate([x_test_encoder, x_test_decoder], x_target_test_decoder)
print('Loss of the test dataset =' + str(round(loss,2)))

model_train.save('seq2seq.h5')


# **Avgerage BLEU Score on char to char decoded strings**
bleu_scores = []
for i in range(0, 510):
    reference = test_target_text[i][1:-1]
    char_reference = list(reference)
    input_seq = x_test_encoder[i:i+1]
    decoded_sentence = decode_sequence(input_seq)
    candidate = decoded_sentence[0:-1]
    char_candidate = list(candidate)
    print('Spanish (true): ', reference)
    print('Spanish (pred): ', candidate)
    # candidate = model_train.predict([x_test_encoder[i:i+1], x_test_decoder[i:i+1]])[0]
    bleu_score = sentence_bleu(char_reference, char_candidate, weights=(1, 0, 0, 0))
    print(bleu_score)
    bleu_scores.append(bleu_score)

average_BS = sum(bleu_scores)/ len(bleu_scores)
print('Average Bleu Score for test dataset:', str(round(average_BS,2)))


# In Section 5 , I have taken 60000 datasets from which 40000 is
#training dataset, 10000 validation dataset and 10000 testing dataset.
#In addition to that, I have implemented multinomial sampling while decoding
#the sequences and used word to index dict fro the decoder and encoder test
#dataset. Apart from that, I have used Bleu score for model quality prediction
#with the 1 gram weight implementation as char to char sentence comparisions
#are being. Finally, I have implied bleu score on 510 test dataset as it was
#not possible to make run for whole 10000 dataset. However, I have tried improve the BLeu score which is approximately between 3.2 to 4.0 and even the loss has been reduced a lot more then expected.
