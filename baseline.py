import numpy as np
np.random.seed(0)
# p = np.random.permutation(300) # n_samples = 108

import matplotlib.pyplot as plt

import pandas as pd
import os, math, glob, time

from sklearn.model_selection import KFold

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l1, l2
import tensorflow.keras.backend as K

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

def prepare_data(condition=False):
	dataset_dir = '../CIS'

	# data_files = sorted(glob.glob(os.path.join(dataset_dir, 'training_data/*.csv')))
	label_file = glob.glob(os.path.join(dataset_dir, 'data_labels/CIS-PD_Training_Data_IDs_Labels.csv'))[0]
	labels_df = pd.read_csv(label_file)
	print(labels_df.shape)
	labels_df = labels_df.loc[np.logical_not(np.isnan(labels_df['on_off']))]
	print(labels_df.shape)
	labels_df = labels_df.loc[np.logical_not(np.isnan(labels_df['tremor']))]
	print(labels_df.shape)

	subject_ids, counts = np.unique(labels_df['subject_id'], return_counts=True)

	measurement_ids, on_off_labels, dyskinesia_labels, tremor_labels = [], [], [], []
	for subject_id in subject_ids:
		df = labels_df.loc[labels_df['subject_id']==subject_id]
		measurement_ids.append(list(df['measurement_id']))
		on_off_labels.append(list(df['on_off']))
		dyskinesia_labels.append(list(df['dyskinesia']))
		tremor_labels.append(list(df['tremor']))

	all_n_data, all_data = [], []
	for idx, subject_id in enumerate(subject_ids):
		subject_files, subject_data, subject_n_data = [], [], []
		# times = []
		for jdx, measurement_id in enumerate(measurement_ids[idx]):
			file = glob.glob(os.path.join(dataset_dir, 'training_data/{}.csv'.format(measurement_id)))[0]
			subject_files.append(file)
			# times.append(np.array(pd.read_csv(file))[-1,0]/60)
			data = np.delete(np.array(pd.read_csv(file)), 0, 1)
			# if condition==True:
			# 	[on_off_labels[idx][jdx]]*
			subject_data.append(data)
			n_data = data.shape[0]
			subject_n_data.append(n_data)
		all_data.append(subject_data)
		all_n_data.append(subject_n_data)

	print('\n..... Data loaded\n')

	return subject_ids, measurement_ids, all_data, all_n_data, on_off_labels, dyskinesia_labels, tremor_labels

def baseline(which='all'):

	# Total 16 subjects

	################################################## ON-OFF
	# 15 subjects
	# Subject 1046 has all NA
	# Achieved a min score of 1.454 with 1.078

	################################################## DYSKINESIA
	# 11 subjects
	# Subject 1006, 1020, 1032, 1046, 1051 has all NA
	# Achieved a min score of 0.997 with 0.822	

	################################################## TREMOR
	# 13 subjects
	# Subject 1039, 1044, 1051 has all NA
	# Achieved a min score of 0.894 with 1.015

	subject_ids, _, _, _, on_off_labels, dyskinesia_labels, tremor_labels = prepare_data()
	
	if which == 'all' or 'on_off':
		final_scores = []
		print('\n------------------ Trying ON_OFF ------------------')
		flag = 0
		for trial in np.arange(0, 4.001, 0.001):
			mse_values, n_values = [], []
			numerator, denominator = 0, 0
			for idx, label in enumerate(on_off_labels):	
				raw = list(map(lambda x: (x-trial)**2, filter(lambda i: not math.isnan(i), label)))
				if len(raw)==0: # skip this loop and continue
					if flag==0: # prints only once
						print('Subject {} has all NA'.format(subject_ids[idx]))
					continue
				mse = np.mean(raw)
				mse_values.append(mse)
				n_values.append(len(raw))
			for mse, n in zip(mse_values, n_values):
				numerator += (mse*math.sqrt(n))
				denominator += math.sqrt(n)
			final_score = numerator/denominator
			final_scores.append(final_score)
			flag = 1
		print('Achieved a min score of {} with {}'.format(np.min(final_scores), np.arange(0, 4.001, 0.001)[np.argmin(final_scores)]))
		print(len(subject_ids), len(mse_values), len(n_values))

	if which == 'all' or 'dyskinesia':
		final_scores = []
		print('\n------------------ Trying Dyskinesia ------------------')
		flag = 0
		for trial in np.arange(0, 4.001, 0.001):
			mse_values, n_values = [], []
			numerator, denominator = 0, 0
			for idx, label in enumerate(dyskinesia_labels):	
				raw = list(map(lambda x: (x-trial)**2, filter(lambda i: not math.isnan(i), label)))
				if len(raw)==0: # skip this loop and continue
					if flag==0: # prints only once
						print('Subject {} has all NA'.format(subject_ids[idx]))
					continue
				mse = np.mean(raw)
				mse_values.append(mse)
				n_values.append(len(raw))
			for mse, n in zip(mse_values, n_values):
				numerator += (mse*math.sqrt(n))
				denominator += math.sqrt(n)
			final_score = numerator/denominator
			final_scores.append(final_score)
			flag = 1
		print('Achieved a min score of {} with {}'.format(np.min(final_scores), np.arange(0, 4.001, 0.001)[np.argmin(final_scores)]))
		print(len(subject_ids), len(mse_values), len(n_values))

	if which == 'all' or 'tremor':
		final_scores = []
		print('\n------------------ Trying Tremor ------------------')
		flag = 0
		for trial in np.arange(0, 4.001, 0.001):
			mse_values, n_values = [], []
			numerator, denominator = 0, 0
			for idx, label in enumerate(tremor_labels):	
				raw = list(map(lambda x: (x-trial)**2, filter(lambda i: not math.isnan(i), label)))
				if len(raw)==0: # skip this loop and continue
					if flag==0: # prints only once
						print('Subject {} has all NA'.format(subject_ids[idx]))
					continue
				mse = np.mean(raw)
				mse_values.append(mse)
				n_values.append(len(raw))
			for mse, n in zip(mse_values, n_values):
				numerator += (mse*math.sqrt(n))
				denominator += math.sqrt(n)
			final_score = numerator/denominator
			final_scores.append(final_score)
			flag = 1
		print('Achieved a min score of {} with {}'.format(np.min(final_scores), np.arange(0, 4.001, 0.001)[np.argmin(final_scores)]))
		print(len(subject_ids), len(mse_values), len(n_values))

# def clean_na(all_data, labels):

# 	X, y = [], []
# 	# X ~ (subjects, datapoints, timesteps, 3)
# 	for idx, subject_label in enumerate(labels):
# 		# X_subject is a list of np arrays shape (timesteps, 3)
# 		X_subject, y_subject = [], []
# 		for jdx, label in enumerate(subject_label):
# 			if not math.isnan(label):
# 				y_subject.append(np.float32(label))
# 				X_subject.append(all_data[idx][jdx].astype(np.float32))
# 		if len(X_subject)!=0:	X.append(X_subject)
# 		if len(y_subject)!=0:	y.append(y_subject)

# 	return X, y

def create_model(input_shape):
	input_data = Input(shape=input_shape)
	x = Conv1D(16, 7, strides=3, padding='valid', activation='relu')(input_data)
	x = Conv1D(16, 7, strides=3, padding='valid', activation='relu')(x)
	x = MaxPool1D()(x)
	x = Conv1D(32, 5, strides=2, padding='valid', activation='relu')(x)
	x = Conv1D(32, 5, strides=2, padding='valid', activation='relu')(x)
	x = MaxPool1D()(x)
	x = Conv1D(64, 3, strides=1, padding='valid', activation='relu')(x)
	x = Conv1D(64, 3, strides=1, padding='valid', activation='relu')(x)
	x = MaxPool1D()(x)
	x = Conv1D(128, 3, strides=1, padding='valid', activation='relu')(x)
	x = Conv1D(128, 3, strides=1, padding='valid', activation='relu')(x)
	x = MaxPool1D()(x)
	x = Conv1D(256, 3, strides=1, padding='valid', activation='relu')(x)
	x = Conv1D(256, 3, strides=1, padding='valid', activation='relu')(x)
	x = GlobalMaxPool1D()(x)
	x = K.expand_dims(x, axis=1)
	# x = Lambda(lambda x: tf.reduce_max(x, axis=1))(x)
	x = Conv1D(1, 1, strides=1, padding='valid', activation='relu')(x)
	model = Model(input_data, x)
	return model

def pad(x, pad_value=1, length='max'): # per subject, to be called for every subject
	
	if length=='max':
		length = max(list(map(lambda i: i.shape[0], x)))

	left_pad_length = lambda i: round((length-len(i))*0.5)
	right_pad_length = lambda i: length - round((length-len(i))*0.5) - len(i)
	padded = list(map(lambda i: np.pad(i, ((left_pad_length(i),right_pad_length(i)), (0,0)), 'constant', constant_values=(pad_value,)), x))

	return np.array(padded)

def training(X, Y, folds=5):

	for index, x in enumerate(X): # iterating over subjects

		x = pad(x).astype(np.float32)
		y = np.array(Y[index]).astype(np.float32)

		print('#'*50)
		print(x.shape)
		print(len(y))
		print()

		fold = 0
		epochs = 5000
		batch_size = 8 # try 4
		learning_rate = 0.001

		model_dir = 'models'
		timeString = time.strftime("%Y%m%d-%H%M%S", time.localtime())
		log_name = "{}".format(timeString)
		log_name = "pad-5"
		train_scores, val_scores = [], []

		for train_index, val_index in KFold(folds).split(x):

			fold+=1
			x_train, x_val = x[train_index], x[val_index]
			y_train, y_val = y[train_index], y[val_index]
			
			model = create_model(input_shape=x_train.shape[1:])

			model.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
			 loss=tf.keras.losses.mean_squared_error)

			checkpointer = tf.keras.callbacks.ModelCheckpoint(
							os.path.join(model_dir, str(subject_ids[index]), '{}.h5'.format(fold)), monitor='loss', verbose=0, save_best_only=True,
							save_weights_only=False, mode='auto', save_freq='epoch')

			tensorboard = tf.keras.callbacks.TensorBoard(log_dir="logs/{}***{}".format(log_name, fold), histogram_freq=1, write_graph=True, write_images=False)

			model.fit(x_train, y_train,
							batch_size=batch_size,
							epochs=epochs,
							verbose=1,
							callbacks=[checkpointer, tensorboard],
							validation_data=(x_val, y_val))

			model = tf.keras.models.load_model(os.path.join(model_dir, str(subject_ids[index]), '{}.h5'.format(fold)))
			
			print('{:2f}'.format(model.predict(x_train)))
			print()
			print('****** Y train ******')
			print('{:2f}'.format(y_train))

			train_score = math.sqrt(model.evaluate(x_train, y_train, verbose=0))
			train_scores.append(train_score)
			val_score = math.sqrt(model.evaluate(x_val, y_val, verbose=0))
			val_scores.append(val_score)

			print()
			print('Train score:', train_score)
			print('Train score mean till fold {} is {}'.format(fold, np.mean(train_scores)))
			print('Val score:', val_score)
			print('Val score mean till fold {} is {}'.format(fold, np.mean(val_scores)))
			print()
			exit()

		print(train_scores)
		print(np.min(train_scores), np.mean(train_scores), np.max(train_scores))
		print()
		print(val_scores)
		print(np.min(val_scores), np.mean(val_scores), np.max(val_scores))	    
		
		exit()


# model = create_model(input_shape=(60000,3))
# print(model.summary())

# baseline()
subject_ids, _, all_data, _, on_off_labels, dyskinesia_labels, tremor_labels = prepare_data()
training(all_data, tremor_labels)

# for subject_labels in tremor_labels:
# 	count = {
# 	'0': subject_labels.count(0),
# 	'1': subject_labels.count(1),
# 	'2': subject_labels.count(2),
# 	'3': subject_labels.count(3),
# 	'4': subject_labels.count(4)
# 	}
# 	print(count)
# colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'lime', 'darkred', 'grey', 'orange', 'gold']
# for idx, data in enumerate(all_data):
# 	lengths = []
# 	for d in data:
# 		lengths.append(d.shape[0])
# 	plt.plot(lengths, c=colors[idx], label=subject_ids[idx])
# 	print(subject_ids[idx], ':', len(data), np.min(lengths), np.mean(lengths), np.max(lengths))
# plt.legend()
# plt.show()

'''
tremor
(1372, 5)
12 12

subject : measurement_files, np.min(lengths), np.mean(lengths), np.max(lengths)
1004 : 82 6888 56326.57 61576
1006 : 37 10720 58296.72 59999
1007 : 276 8464 57484.65 60085
1019 : 45 25065 58641.28 59587
1020 : 195 6155 57695.13 61224
1023 : 106 14947 57175.08 59922
1032 : 177 5803 58097.89 59884
1034 : 40 29627 58843.55 60011
1038 : 207 9495 58718.07 61960
1043 : 34 19789 54661.91 59732
1048 : 91 9909 54757.80 60309
1049 : 82 15847 57353.90 60457

{'0': 41, '1': 16, '2': 10, '3': 14, '4': 1}
{'0': 12, '1': 18, '2': 7, '3': 0, '4': 0}
{'0': 190, '1': 77, '2': 9, '3': 0, '4': 0}
{'0': 7, '1': 24, '2': 14, '3': 0, '4': 0}
{'0': 15, '1': 156, '2': 24, '3': 0, '4': 0}
{'0': 74, '1': 28, '2': 3, '3': 1, '4': 0}
{'0': 105, '1': 70, '2': 2, '3': 0, '4': 0}
{'0': 20, '1': 19, '2': 1, '3': 0, '4': 0}
{'0': 1, '1': 134, '2': 71, '3': 1, '4': 0}
{'0': 12, '1': 8, '2': 7, '3': 7, '4': 0}
{'0': 0, '1': 1, '2': 22, '3': 60, '4': 8}
{'0': 2, '1': 30, '2': 31, '3': 19, '4': 0}

Doesnt train: pad_value 5, 
'''

'''
dyskinesia
(1165, 5)
11 11

82 82
276 276
45 45
106 106
40 40
207 207
130 130
34 34
72 72
91 91
82 82
'''


