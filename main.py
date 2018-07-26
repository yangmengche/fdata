# from pandas import load_csv
import pandas as pd
import numpy as np

ICT_SIZE = 15
# FCT_SIZE = 153
FCT_SIZE = 90
EPOCHS=100
BATCH_SIZE=64

TRAINING_FILE='/2018-07-12/t3-train-3000-25-25-25-25.csv'
TESTING_FILE='/2018-07-12/t3-test-1000-25-25-25-25.csv'

# return [ict, fct, lable]
def load_csv(fileName):
  dataFrame = pd.read_csv(fileName, header=0)
  #shape ict=(1000, 15), fct=(1000, 153), lable=(1000, 2)
  return np.split(dataFrame.values, [ICT_SIZE, ICT_SIZE+FCT_SIZE, ICT_SIZE+FCT_SIZE+1], axis=1)

[ict, fct, label, reason]=load_csv('./data/'+TRAINING_FILE)
[ict_test, fct_test, label_test, reason_test]=load_csv('./data/'+TESTING_FILE)

def preProcess(ict, fct, label):
  for i, l in enumerate(label):
    label[i]= 1 if l ==3 else 0

  label = label.flatten()
  # splite data to training, cv, testing
  # [X_training_ict, X_testing_ict] = np.split(ict, [TRAINING_SIZE], axis=0)
  # [X_training_fct, X_testing_fct] = np.split(fct, [TRAINING_SIZE], axis=0)
  # [y_training, y_testing] = np.split(label, [TRAINING_SIZE], axis=0)

  # standardization
  ict_mean = ict.mean(axis=0)
  ict -= ict_mean
  ict_std = ict.std(axis=0, dtype='float64')
  ict /= ict_std

  fct_mean = fct.mean(axis=0)
  fct -= fct_mean
  fct_std = fct.std(axis=0, dtype='float64')
  fct /= fct_std
  return ict, fct, label

X_training_ict, X_training_fct, y_training = preProcess(ict, fct, label)
TRAINING_SIZE = label.shape[0]
X_testing_ict, X_testing_fct, y_testing = preProcess(ict_test, fct_test, label_test)
TESTING_SIZE = label_test.shape[0]

from keras.layers import Input, Dense, SimpleRNN, Conv1D, MaxPooling1D, concatenate, Reshape, regularizers
from keras.models import Model, Sequential
from keras.preprocessing import sequence
import math

def predict_result(output, label):
  correct=0
  for i, r in enumerate(output):
    if (r[0] >= 0.5 and label[i] == 1) or (r[0] < 0.5 and label[i] == 0):
      correct += 1
  print('correct={0}/total={1}'.format(correct, TESTING_SIZE))
  return correct / TESTING_SIZE

def dnn():
  print('\n*** DNN ***\n')
  X_training = np.concatenate((X_training_ict, X_training_fct), axis=1)
  X_testing = np.concatenate((X_testing_ict, X_testing_fct), axis=1)
  model=Sequential()
  model.add(Dense(32, activation='relu', input_shape=(ICT_SIZE+FCT_SIZE,), kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l2(0.01)))
  # model.add(Dense(8, activation='relu'))
  # model.add(Dense(25, activation='relu'))
  model.add(Dense(1))
  model.compile(optimizer='rmsprop', loss='mse', metrics=['accuracy'])
  model.summary()

  t1 = time.time()
  history = model.fit(X_training, y_training, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.2)
  t = time.time() - t1
  print('total training time={0:.2f} sec'.format(t))

  t1 = time.time()
  output = model.evaluate(X_testing, y_testing)
  t = time.time() - t1
  print('evaluation={0}, time={1:.2f}'.format(output[1], t))

  # t1 = time.time()
  # output = model.predict(X_testing)
  # t = time.time() - t1
  # result = predict_result(output, y_testing)
  # print('predict={0}, predict time={1:.2f}'.format(result, t))
  return model, history

def dnn_part():
  print('\n*** DNN Part***\n')
  model=Sequential()
  model.add(Dense(64, activation='relu', input_shape=(FCT_SIZE,), kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l2(0.01)))
  # model.add(Dense(64, activation='relu'))
  # model.add(Dense(21, activation='sigmoid'))
  model.add(Dense(1, activation='sigmoid'))
  model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
  model.summary()
  history = model.fit(X_training_fct, y_training, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.2)
  return model, history

def simple_rnn():
  print('\n*** Simple RNN ***\n')
  pad = np.zeros((TRAINING_SIZE, FCT_SIZE - ICT_SIZE))
  temp = np.concatenate((X_training_ict, pad), axis=1)
  X_training = np.concatenate((temp, X_training_fct), axis=1)
  X_training = X_training.reshape(TRAINING_SIZE, 2, FCT_SIZE)

  pad = np.zeros((TESTING_SIZE, FCT_SIZE - ICT_SIZE))
  temp = np.concatenate((X_testing_ict, pad), axis=1)
  X_testing = np.concatenate((temp, X_testing_fct), axis=1)
  X_testing = X_testing.reshape(TESTING_SIZE, 2, FCT_SIZE)

  model=Sequential()
  model.add(SimpleRNN(64, input_shape=(2, FCT_SIZE), activation='relu', return_sequences=False, stateful=False, dropout=0.2))
  # model.add(Dense(16))
  model.add(Dense(1, name='main_output', activation='sigmoid'))
  model.compile(optimizer='rmsprop', loss='mse', metrics=['accuracy'])
  model.summary()

  t1 = time.time()
  history = model.fit(X_training, y_training, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.2)
  t = time.time() - t1
  print('total training time={0:.2f} sec'.format(t))

  t1 = time.time()
  output = model.evaluate(X_testing, y_testing)
  t = time.time() - t1
  print('evaluation={0}, time={1:.2f}'.format(output[1], t))

  # t1 = time.time()
  # output = model.predict(X_testing)
  # t = time.time() - t1
  # result = predict_result(output, y_testing)
  # print('predict={0}, predict time={1:.2f}'.format(result, t))  
  return model, history

def dense_rnn():
  print('\n*** DENSE RNN ***\n')
  DENSE_SIZE=15
  ict = Input(shape=(ICT_SIZE,), name='ict')
  d1 = Dense(DENSE_SIZE, activation='relu', kernel_regularizer=regularizers.l2(0.0), activity_regularizer=regularizers.l2(0.0))(ict)
  fct = Input(shape=(FCT_SIZE,), name='fct')
  d2 = Dense(DENSE_SIZE, activation='relu', kernel_regularizer=regularizers.l2(0.0), activity_regularizer=regularizers.l2(0.0))(fct)

  x = concatenate([d1, d2])
  x = Reshape((2, DENSE_SIZE), input_shape=(1, DENSE_SIZE*2))(x)
  x = SimpleRNN(64, input_shape=(None, 2, DENSE_SIZE), return_sequences=False, stateful=False, dropout=0.2)(x)
  output = Dense(1, name='main_output', activation='sigmoid')(x)

  model = Model(input=[ict, fct], output=output)

  model.compile(optimizer='rmsprop', loss='mse', metrics=['accuracy'])
  model.summary()

  t1 = time.time()
  history = model.fit({'ict': X_training_ict, 'fct':X_training_fct}, y_training, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.2)
  t = time.time() - t1
  print('total training time={0:.2f} sec'.format(t))

  t1 = time.time()
  output = model.evaluate({'ict': X_testing_ict, 'fct':X_testing_fct}, y_testing)
  t = time.time() - t1
  print('evaluation={0}, time={1:.2f}'.format(output[1], t))

  # t1 = time.time()
  # output = model.predict({'ict': X_testing_ict, 'fct':X_testing_fct})
  # t = time.time() - t1
  # result = predict_result(output, y_testing)
  # print('predict={0}, predict time={1:.2f}'.format(result, t))
  return model, history

def cnn_rnn_15_153():
  print('\n*** CNN RNN 15+153 ***\n')
  CONV_SIZE=6
  KERNEL_SIZE=3
  PADDING_SIZE=4
  ICT_PADDING_SIZE=ICT_SIZE+PADDING_SIZE  
  pad = np.zeros((TRAINING_SIZE, PADDING_SIZE))
  ict_padding = np.concatenate((X_training_ict, pad), axis=1)

  pad = np.zeros((TESTING_SIZE, PADDING_SIZE))
  ict_test_padding = np.concatenate((X_testing_ict, pad), axis=1)  

  ict = Input(shape=(ICT_PADDING_SIZE,), name='ict')
  # (1, 19)
  ict_reshape = Reshape((ICT_PADDING_SIZE, 1), input_shape=(None, 1, ICT_PADDING_SIZE))(ict)
  # (19, 1)
  # ict max_length=ICT_SIZE, input_dim=1
  c1 = Conv1D(CONV_SIZE, KERNEL_SIZE, activation='relu', input_shape=(ICT_PADDING_SIZE, 1))(ict_reshape)
  # (17, 2)
  c1 = Reshape((1, (ICT_PADDING_SIZE-KERNEL_SIZE+1)*CONV_SIZE), input_shape=(ICT_PADDING_SIZE-KERNEL_SIZE+1, CONV_SIZE))(c1)
  # (1, 34)

  fct = Input(shape=(FCT_SIZE,), name='fct')
  # (1, 153)
  fct_reshape = Reshape((FCT_SIZE, 1), input_shape=(None, 1, FCT_SIZE))(fct)
  # (153, 1)
  input_shape = (FCT_SIZE, 1)
  c2 = Conv1D(CONV_SIZE, KERNEL_SIZE, activation='relu', input_shape=input_shape)(fct_reshape)
  # (151, 2)
  c2 = MaxPooling1D(pool_size=2)(c2)
  # (75, 2)
  input_shape = (math.floor((input_shape[0]-KERNEL_SIZE+1)/2), CONV_SIZE)
  c2 = Conv1D(CONV_SIZE, KERNEL_SIZE, activation='relu', input_shape=input_shape)(c2)
  # (73, 2)
  c2 = MaxPooling1D(pool_size=2)(c2)
  # (36, 2)
  input_shape = (math.floor((input_shape[0]-KERNEL_SIZE+1)/2), CONV_SIZE)
  c2 = Conv1D(CONV_SIZE, KERNEL_SIZE, activation='relu', input_shape=input_shape)(c2)
  # (34, 2)
  c2 = MaxPooling1D(pool_size=2)(c2)  
  # (17, 2)
  input_shape = (math.floor((input_shape[0]-KERNEL_SIZE+1)/2), CONV_SIZE)
  c2 = Reshape((1, input_shape[0]*CONV_SIZE), input_shape=input_shape)(c2)
  # (1, 34)
 
  x = concatenate([c1, c2])
  # (1, 68)
  RNN_UNIT_SIZE=int(input_shape[0]*CONV_SIZE)
  x = Reshape((2, RNN_UNIT_SIZE), input_shape=(1, RNN_UNIT_SIZE*2))(x)

  x = SimpleRNN(32, input_shape=(None, 2, RNN_UNIT_SIZE), return_sequences=False, stateful=False, dropout=0.2)(x)
  output = Dense(1, name='main_output', activation='sigmoid')(x)

  model = Model(input=[ict, fct], output=output)

  model.compile(optimizer='rmsprop', loss='mse', metrics=['accuracy'])
  model.summary()

  t1 = time.time()
  history = model.fit({'ict': ict_padding, 'fct':X_training_fct}, y_training, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.2)
  t = time.time() - t1
  print('total training time={0:.2f} sec'.format(t))

  t1 = time.time()
  output = model.evaluate({'ict': ict_padding, 'fct':X_testing_fct}, y_testing)
  t = time.time() - t1
  print('evaluation={0}, time={1:.2f}'.format(output[1], t))

  # t1 = time.time()
  # output = model.predict({'ict': ict_test_padding, 'fct':X_testing_fct})
  # t = time.time() - t1
  # result = predict_result(output, y_testing)
  # print('predict={0}, predict time={1:.2f}'.format(result, t))
  return model, history

def cnn_rnn_15_90():
  print('\n*** CNN RNN 15+90 ***\n')
  CONV_SIZE=6
  KERNEL_SIZE=3
  PADDING_SIZE=6
  ICT_PADDING_SIZE=ICT_SIZE+PADDING_SIZE
  pad = np.zeros((TRAINING_SIZE, PADDING_SIZE))
  ict_padding = np.concatenate((X_training_ict, pad), axis=1)

  pad = np.zeros((TESTING_SIZE, PADDING_SIZE))
  ict_test_padding = np.concatenate((X_testing_ict, pad), axis=1)  

  ict = Input(shape=(ICT_PADDING_SIZE,), name='ict')
  # ex: CONV_SIZE=2, KERNEL_SIZE=3
  # ex: (1, 21)
  ict_reshape = Reshape((ICT_PADDING_SIZE, 1), input_shape=(None, 1, ICT_PADDING_SIZE))(ict)
  # ex: (21, 1)
  # ict max_length=ICT_SIZE, input_dim=1
  c1 = Conv1D(CONV_SIZE, KERNEL_SIZE, activation='relu', input_shape=(ICT_PADDING_SIZE, 1))(ict_reshape)
  # ex: (19, 2)
  c1_reshape = Reshape((1, (ICT_PADDING_SIZE-KERNEL_SIZE+1)*CONV_SIZE), input_shape=(ICT_PADDING_SIZE-KERNEL_SIZE+1, CONV_SIZE))(c1)
  # ex: (1, 38)

  fct = Input(shape=(FCT_SIZE,), name='fct')
  # ex: (1, 90)
  fct_reshape = Reshape((FCT_SIZE, 1), input_shape=(None, 1, FCT_SIZE))(fct)
  # ex: (90, 1)
  input_shape = (FCT_SIZE, 1)
  c2 = Conv1D(CONV_SIZE, KERNEL_SIZE, activation='relu', input_shape=input_shape)(fct_reshape)
  # ex: (88, 2)
  c2_pooling = MaxPooling1D(pool_size=2)(c2)
  # ex:(44, 2)
  input_shape = (math.floor((input_shape[0]-KERNEL_SIZE+1)/2), CONV_SIZE)
  c3 = Conv1D(CONV_SIZE, KERNEL_SIZE, activation='relu', input_shape=input_shape)(c2_pooling)
  # ex:(42, 2)
  c3_pooling = MaxPooling1D(pool_size=2)(c3)
  # ex: (21, 2)
  input_shape = (math.floor((input_shape[0]-KERNEL_SIZE+1)/2), CONV_SIZE)
  c4 = Conv1D(CONV_SIZE, KERNEL_SIZE, activation='relu', input_shape=input_shape)(c3_pooling)
  # ex: (19, 2)

  input_shape = (input_shape[0]-KERNEL_SIZE+1, CONV_SIZE)
  c4_reshape = Reshape((1, input_shape[0]*CONV_SIZE), input_shape=input_shape)(c4)
  # ex: (1, 38)
 
  x = concatenate([c1_reshape, c4_reshape])
  # ex: (1, 76)
  RNN_UNIT_SIZE=int(input_shape[0]*CONV_SIZE)
  x_reshape = Reshape((2, RNN_UNIT_SIZE), input_shape=(1, RNN_UNIT_SIZE*2))(x)

  rnn = SimpleRNN(64, input_shape=(None, 2, RNN_UNIT_SIZE), return_sequences=False, stateful=False, dropout=0.1)(x_reshape)
  output = Dense(1, name='main_output', activation='sigmoid')(rnn)

  model = Model(input=[ict, fct], output=output)

  model.compile(optimizer='rmsprop', loss='mse', metrics=['accuracy'])
  model.summary()

  t1 = time.time()
  history = model.fit({'ict': ict_padding, 'fct':X_training_fct}, y_training, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.2)
  t = time.time() - t1
  print('total training time={0:.2f} sec'.format(t))

  t1 = time.time()
  output = model.evaluate({'ict': ict_test_padding, 'fct':X_testing_fct}, y_testing)
  t = time.time() - t1
  print('evaluation={0}, time={1:.2f}'.format(output[1], t))

  # t1 = time.time()
  # output = model.predict({'ict': ict_test_padding, 'fct':X_testing_fct})
  # t = time.time() - t1
  # result = predict_result(output, y_testing)
  # print('predict={0}, predict time={1:.2f}'.format(result, t))
  return model, history

def plot(history):
  import matplotlib.pyplot as plt
  loss = history.history['loss']
  val_loss = history.history['val_loss']
  acc = history.history['acc']
  val_acc = history.history['val_acc']
  epochs = range(1, len(loss) + 1)
  plt.subplot(2, 1, 1)
  plt.plot(epochs, loss, 'b', label='Training loss')
  plt.plot(epochs, val_loss, 'r', label='Validation loss')
  plt.title('Training and validation loss')
  plt.xlabel('Epochs')
  plt.ylabel('Loss')
  plt.legend()
  plt.subplot(2, 1, 2)
  plt.plot(epochs, acc, 'b', label='Training accuracy')
  plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
  plt.ylabel('accuracy')
  plt.legend()
  plt.show()

import time


# model, history = dnn()
# model, history = simple_rnn()
model, history = dense_rnn()
# model, history = cnn_rnn_15_153()
# model, history = cnn_rnn_15_90()

plot(history)


