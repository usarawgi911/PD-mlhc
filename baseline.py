import numpy as np
np.random.seed(0)
# p = np.random.permutation(300) # n_samples = 108

import matplotlib.pyplot as plt

import pandas as pd
import os, math, glob, time

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l1, l2
import tensorflow.keras.backend as K

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="7"

def prepare_data(condition=False):
	'''
	Saves data in .npy files to avoid date parsing delays in every run
	condition = True allows medication status to be used as an input feature for modelling drug response / treatment effect 
	'''

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
			if condition==True:
				treatment = np.expand_dims(np.array([on_off_labels[idx][jdx]]*data.shape[0]), axis=-1)
				data = np.concatenate((data, treatment), axis=-1)
			subject_data.append(data)
			n_data = data.shape[0]
			subject_n_data.append(n_data)
		all_data.append(subject_data)
		all_n_data.append(subject_n_data)

	print('\n..... Data loaded\n')

	np.save('../CIS/subject_ids.npy', np.array(subject_ids))
	np.save('../CIS/measurement_ids.npy', np.array(measurement_ids))
	np.save('../CIS/all_data.npy', np.array(all_data))
	np.save('../CIS/all_n_data.npy', np.array(all_n_data))
	np.save('../CIS/on_off_labels.npy', np.array(on_off_labels))
	np.save('../CIS/dyskinesia_labels.npy', np.array(dyskinesia_labels))
	np.save('../CIS/tremor_labels.npy', np.array(tremor_labels))

	print('\n..... Data saved\n')

	return subject_ids, measurement_ids, all_data, all_n_data, on_off_labels, dyskinesia_labels, tremor_labels

def load_data():
	'''
	Loads data (unpadded) from .npy files 
	'''

	subject_ids = np.load('../CIS/subject_ids.npy', allow_pickle=True)
	measurement_ids = np.load('../CIS/measurement_ids.npy', allow_pickle=True)
	all_data = np.load('../CIS/all_data.npy', allow_pickle=True)
	all_n_data = np.load('../CIS/all_n_data.npy', allow_pickle=True)
	on_off_labels = np.load('../CIS/on_off_labels.npy', allow_pickle=True)
	dyskinesia_labels = np.load('../CIS/dyskinesia_labels.npy', allow_pickle=True)
	tremor_labels = np.load('../CIS/tremor_labels.npy', allow_pickle=True)

	print('\n..... Data loaded from numpy files\n')

	return subject_ids, measurement_ids, all_data, all_n_data, on_off_labels, dyskinesia_labels, tremor_labels

def new_baseline():

	rmse_scores, random_preds = [], []
	for i, subject in enumerate(subject_ids):
		rmse_trials, trials = [], []
		for trial in np.arange(0, 4.001, 0.001):
			y_true = tremor_labels[i]
			rmse = mean_squared_error(y_true, [trial]*len(y_true), squared=False)
			rmse_trials.append(rmse)
			trials.append(trial)
		idx = np.argmin(rmse_trials)
		subject_rmse, subject_trial = rmse_trials[idx], trials[idx]
		rmse_scores.append(subject_rmse)
		random_preds.append(subject_trial)

	for i, subject in enumerate(subject_ids):
		print('Subject:', subject)
		print('Random Pred: {:.3f}'.format(random_preds[i]))
		print('RMSE score: {:.3f}'.format(rmse_scores[i]))
		print()

	final_weighted_score = sum(list(map(lambda i: len(measurement_ids[i])*rmse_scores[i], range(len(rmse_scores)))))
	final_weighted_score /= sum([len(j) for j in measurement_ids])
	final_root_weighted_score = sum(list(map(lambda i: math.sqrt(len(measurement_ids[i]))*rmse_scores[i], range(len(rmse_scores)))))
	final_root_weighted_score /= sum([math.sqrt(len(j)) for j in measurement_ids])
	print('Final score: {:.3f}'.format(np.mean(rmse_scores)))
	print('Final weighted score: {:.3f}'.format(final_weighted_score))
	print('Final root weighted score: {:.3f}'.format(final_root_weighted_score))


def baseline(which='all'):
	'''
	Gets an RMSE baseline score by random by random constant assignment as prediction 
	'''

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
	'''
	Creates FCN
	'''

	input_data = Input(shape=input_shape)

	x = BatchNormalization()(input_data)

	x = Conv1D(16, 7, strides=3, padding='valid')(x)
	x = LeakyReLU()(x)
	x = Conv1D(16, 7, strides=3, padding='valid')(x)
	x = LeakyReLU()(x)
	x = MaxPool1D()(x)
	x = BatchNormalization()(x)

	x = Conv1D(32, 5, strides=2, padding='valid')(x)
	x = LeakyReLU()(x)
	x = Conv1D(32, 5, strides=2, padding='valid')(x)
	x = LeakyReLU()(x)
	x = MaxPool1D()(x)
	x = BatchNormalization()(x)

	x = Conv1D(64, 3, strides=1, padding='valid')(x)
	x = LeakyReLU()(x)
	x = Conv1D(64, 3, strides=1, padding='valid')(x)
	x = LeakyReLU()(x)
	x = MaxPool1D()(x)
	x = BatchNormalization()(x)

	x = Conv1D(128, 3, strides=1, padding='valid')(x)
	x = LeakyReLU()(x)
	x = Conv1D(128, 3, strides=1, padding='valid')(x)
	x = LeakyReLU()(x)
	x = MaxPool1D()(x)
	x = BatchNormalization()(x)

	x = Conv1D(256, 3, strides=1, padding='valid')(x)
	x = LeakyReLU()(x)
	x = Conv1D(256, 3, strides=1, padding='valid')(x)
	x = LeakyReLU()(x)
	x = GlobalMaxPool1D()(x)
	x = K.expand_dims(x, axis=1)
	x = BatchNormalization()(x)

	x = Conv1D(1, 1, strides=1, padding='valid')(x)
	x = ReLU(max_value=4)(x)

	model = Model(input_data, x)

	return model

def pad(x, pad_value=-2, length='max'):
	'''
	Pads input data per subject, to be called for every subject
	'''
	
	if length=='max':
		length = max(list(map(lambda i: i.shape[0], x)))

	left_pad_length = lambda i: round((length-len(i))*0.5)
	right_pad_length = lambda i: length - round((length-len(i))*0.5) - len(i)
	padded = list(map(lambda i: np.pad(i, ((left_pad_length(i),right_pad_length(i)), (0,0)), 'constant', constant_values=(pad_value,)), x))

	return np.array(padded)

def training(X, Y, folds=5):
	'''
	Performs 5-fold cross-validated training per subject
	Reports 5-fold train and val RMSE scores per subject, as well as final mean and weighted scores 
	'''

	all_train_scores, all_val_scores, all_val_rounded_scores = [], [], []

	for index, x in enumerate(X): # iterating over subjects

		print('\nSubject {} starting\n'.format(str(subject_ids[index])))

		x = pad(x).astype(np.float32)
		y = np.array(Y[index]).astype(np.float32)

		p = np.random.permutation(x.shape[0])
		x, y = x[p], y[p]

		patience = 200
		fold = 0
		epochs = 1000 # 500
		batch_size = 16
		learning_rate = 1e-3

		# LR 1e-5 # 0.56, 0.74 (p) 0.58, 0.7 || 0.53 0.7
		# 0.63, 0.49 (no p) # LR (0.5 * 1e-4) 0.73, 0.35

		model_dir = 'models' # running in tmux 4
		model_dir = 'models-re-re' # running in tmux 1
		timeString = time.strftime("%Y%m%d-%H%M%S", time.localtime())
		log_name = '{}'.format(timeString)
		log_name = '|=|=|'
		train_scores, val_scores, val_rounded_scores = [], [], []

		for train_index, val_index in KFold(folds).split(x):

			fold+=1
			x_train, x_val = x[train_index], x[val_index]
			y_train, y_val = y[train_index], y[val_index]

			print('x train shape', x_train.shape)
			print('x val shape', x_val.shape)

			scalers = {}
			for i in range(x_train.shape[1]):
				scalers[i] = StandardScaler()
				x_train[:, i, :] = scalers[i].fit_transform(x_train[:, i, :]) 

			for i in range(x_val.shape[1]):
				x_val[:, i, :] = scalers[i].transform(x_val[:, i, :]) 
			
			model = create_model(input_shape=x_train.shape[1:])

			model.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
			 loss=tf.keras.losses.mean_squared_error)

			checkpointer = tf.keras.callbacks.ModelCheckpoint(
							os.path.join(model_dir, str(subject_ids[index]), '{}.h5'.format(fold)), monitor='val_loss', verbose=0, save_best_only=True,
							save_weights_only=False, mode='auto', save_freq='epoch')

			tensorboard = tf.keras.callbacks.TensorBoard(log_dir='logs/{}@{}@{}'.format(log_name, str(subject_ids[index]), fold), histogram_freq=1, write_graph=True, write_images=False)
			es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.001, patience=patience, restore_best_weights=True, verbose=0, mode='auto')			
			
			model.fit(x_train, y_train,
							batch_size=batch_size,
							epochs=epochs,
							verbose=1,
							callbacks=[checkpointer, tensorboard, es],
							validation_data=(x_val, y_val))

			es_epoch = es.stopped_epoch
			print(es_epoch)
			if es_epoch==patience:
				raise Exception('Fold {} for subject {} not training'.format(fold, str(subject_ids[index])))

			model = tf.keras.models.load_model(os.path.join(model_dir, str(subject_ids[index]), '{}.h5'.format(fold)))
			
			train_preds = model.predict(x_train)
			val_preds = model.predict(x_val)

			print()
			print('Preds train \t Preds train rounded \t Y train')
			for j, value in enumerate(y_train):
				print('{:3f} \t {} \t\t\t {}'.format(train_preds[j,0,0], round(train_preds[j,0,0]), y_train[j]))
			print()
			print('Preds val \t Preds val rounded \t Y val')
			for j, value in enumerate(y_val):
				print('{:3f} \t {} \t\t\t {}'.format(val_preds[j,0,0], round(val_preds[j,0,0]), y_val[j]))

			train_score = math.sqrt(model.evaluate(x_train, y_train, verbose=0))
			train_scores.append(train_score)
			val_score = math.sqrt(model.evaluate(x_val, y_val, verbose=0))
			val_scores.append(val_score)
			val_rounded_score = mean_squared_error(y_val, np.squeeze(np.around(val_preds), axis=-1), squared=False)
			val_rounded_scores.append(val_rounded_score)

			print('\nTrain score: {:.3f}'.format(train_score))
			print('Val score: {:.3f}'.format(val_score))
			print('Val rounded score: {:.3f}'.format(val_rounded_score))
			print('\nTrain score mean till fold {} is {:.3f}'.format(fold, np.mean(train_scores)))
			print('Val score mean till fold {} is {:.3f}'.format(fold, np.mean(val_scores)))
			print('Val rounded score mean till fold {} is {:.3f}'.format(fold, np.mean(val_rounded_scores)))
			print()

		print([float('{:.3f}'.format(a)) for a in train_scores])
		print('Mean Train score {:.3f}'.format(np.mean(train_scores)))
		print()
		print([float('{:.3f}'.format(a)) for a in val_scores])
		print('Mean Val score {:.3f}'.format(np.mean(val_scores)))	  
		all_train_scores.append(train_scores)
		all_val_scores.append(val_scores)
		all_val_rounded_scores.append(val_rounded_scores)
		print('\nSubject {} done\n'.format(str(subject_ids[index])))

	subject_train_scores, subject_val_scores = [], []
	for i in range(len(all_train_scores)):
		print()
		print('Subject:', subject_ids[i])
		print('Train scores:', [float('{:.3f}'.format(a)) for a in all_train_scores[i]])
		print('Mean: {:.3f}'.format(np.mean(all_train_scores[i])))
		print('Val scores:', [float('{:.3f}'.format(a)) for a in all_val_scores[i]])
		print('Mean: {:.3f}'.format(np.mean(all_val_scores[i])))
		subject_train_scores.append(np.mean(all_train_scores[i]))
		subject_val_scores.append(np.mean(all_val_scores[i]))

	final_train_score = np.mean(subject_train_scores)
	final_val_score = np.mean(subject_val_scores)

	final_train_weighted_score = sum(list(map(lambda i: len(measurement_ids[i])*subject_train_scores[i], range(len(subject_train_scores)))))
	final_train_weighted_score /= sum([len(j) for j in measurement_ids])
	final_val_weighted_score = sum(list(map(lambda i: len(measurement_ids[i])*subject_val_scores[i], range(len(subject_val_scores)))))
	final_val_weighted_score /= sum([len(j) for j in measurement_ids])

	final_train_root_weighted_score = sum(list(map(lambda i: math.sqrt(len(measurement_ids[i]))*subject_train_scores[i], range(len(subject_train_scores)))))
	final_train_root_weighted_score /= sum([math.sqrt(len(j)) for j in measurement_ids])
	final_val_root_weighted_score = sum(list(map(lambda i: math.sqrt(len(measurement_ids[i]))*subject_val_scores[i], range(len(subject_val_scores)))))
	final_val_root_weighted_score /= sum([math.sqrt(len(j)) for j in measurement_ids])

	print()
	print('Final train score: {:.3f}'.format(final_train_score))
	print('Final val score: {:.3f}'.format(final_train_score))
	print('Final weighted train score: {:.3f}'.format(final_train_weighted_score))
	print('Final weighted val score: {:.3f}'.format(final_val_weighted_score))
	print('Final root weighted train score: {:.3f}'.format(final_train_root_weighted_score))
	print('Final root weighted val score: {:.3f}'.format(final_val_root_weighted_score))

	print('Batch size used:', batch_size)		


# model = create_model(input_shape=(60000,3))
# print(model.summary())

# baseline()
# condition = True
# subject_ids, _, all_data, _, on_off_labels, dyskinesia_labels, tremor_labels = prepare_data(condition=condition)
subject_ids, measurement_ids, all_data, _, on_off_labels, dyskinesia_labels, tremor_labels = load_data() # conditioned
# new_baseline()
# exit()

start = time.time()
training(all_data, tremor_labels)
end = time.time()
time_taken = (end - start)/60 # in minutes
print('\nTraining of all subjects took {:.3f} minutes'.format(time_taken))

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
TREMOR
(1372, 5)

total number of subjects 12 12

lengths are the lengths of data (1 length value per meansurement file)
subject : no. of measurement_files, np.min(lengths), np.mean(lengths), np.max(lengths)
1004 : 82	 6888	 56326.57	 61576
1006 : 37	 10720	 58296.72	 59999
1007 : 276	 8464	 57484.65	 60085
1019 : 45	 25065	 58641.28	 59587
1020 : 195	 6155	 57695.13	 61224
1023 : 106	 14947	 57175.08	 59922
1032 : 177	 5803	 58097.89	 59884
1034 : 40	 29627	 58843.55	 60011
1038 : 207	 9495	 58718.07	 61960
1043 : 34	 19789	 54661.91	 59732
1048 : 91	 9909	 54757.80	 60309
1049 : 82	 15847	 57353.90	 60457

subject : {'tremor value': no. of those values}
1004 : {'0': 41, '1': 16, '2': 10, '3': 14, '4': 1}
1006 : {'0': 12, '1': 18, '2': 7, '3': 0, '4': 0}
1007 : {'0': 190, '1': 77, '2': 9, '3': 0, '4': 0}
1019 : {'0': 7, '1': 24, '2': 14, '3': 0, '4': 0}
1020 : {'0': 15, '1': 156, '2': 24, '3': 0, '4': 0}
1023 : {'0': 74, '1': 28, '2': 3, '3': 1, '4': 0}
1032 : {'0': 105, '1': 70, '2': 2, '3': 0, '4': 0}
1034 : {'0': 20, '1': 19, '2': 1, '3': 0, '4': 0}
1038 : {'0': 1, '1': 134, '2': 71, '3': 1, '4': 0}
1043 : {'0': 12, '1': 8, '2': 7, '3': 7, '4': 0}
1048 : {'0': 0, '1': 1, '2': 22, '3': 60, '4': 8}
1049 : {'0': 2, '1': 30, '2': 31, '3': 19, '4': 0}

Doesnt train: pad_value 5, 

[0.3394637920725939, 0.4262521104095624, 0.8052916631013518, 0.7044685858802354, 0.6316547413596474]
Mean Train score 0.581

[0.7937870407472961, 0.7827336909356533, 0.27961372862392486, 0.8021018699056347, 0.7067172791767017]
Mean Val score 0.673

[0.9546467547250613, 0.6285097376782999, 0.8514569411308717, 0.8576657955532441, 0.6575363784503172]
Mean Train score 0.790

[0.7195390433718822, 0.6785105896687712, 0.25700103327064755, 0.7211952906706071, 0.6336239216084728]
Mean Val score 0.602
'''

'''
All subjects:
Subject: 1004
Train scores: [0.8566905872058075, 0.21877878439863352, 0.6454282069805571, 0.6318724471948138, 0.7637957484518381]  0.623
Val scores: [0.6802666487726059, 0.8151696070197728, 0.4141714384615129, 0.8221341931915283, 0.5963103404592239]     0.666

Subject: 1006
Train scores: [0.12254441319835464, 0.19000757825005785, 0.49816110834277605, 0.7104358370330159, 0.08236934323399339]        0.321
Val scores: [0.3804777727878387, 0.43902287083770897, 0.25308194971747877, 0.5157439643721196, 0.5608803159420591]   0.430

Subject: 1007
Train scores: [0.36523563883650395, 0.18999483134387068, 0.25628403561112933, 0.3413076941974711, 0.1584456933169091] 0.262
Val scores: [0.4797654248825388, 0.3670477237693005, 0.4369531641644632, 0.416410730211938, 0.326341719370683]  0.405

Subject: 1019
Train scores: [0.2130372561262085, 0.462680783841437, 0.34213567502921866, 0.47213728010648354, 0.5014804011012459]  0.398
Val scores: [0.4308523168876289, 0.3671198031739875, 0.28252188203257556, 0.2339523478195539, 0.35512093935306227]   0.334

Subject: 1020
Train scores: [0.17972681841738708, 0.18727775310308023, 0.1282633449044453, 0.11999357306946662, 0.19328258341131282]        0.162
Val scores: [0.3336659301999933, 0.3446600924662855, 0.30141788356228366, 0.44809025705873007, 0.34855002075239083]  0.355

Subject: 1023
Train scores: [0.23340235154114883, 0.498258980505356, 0.20828810924417884, 0.44237469520943223, 0.6076915161964602] 0.398
Val scores: [0.5403002474682439, 0.6102825888762076, 0.2514105432640489, 0.4595755969520688, 0.45389150047303173]    0.463

Subject: 1032
Train scores: [0.1290833043982635, 0.1805536051662032, 0.5922438577656509, 0.2009461786633101, 0.2678026050114706]   0.274
Val scores: [0.49831499183991107, 0.5458624867183647, 0.5336503098429882, 0.4690125097049199, 0.5143766125339381]    0.512

Subject: 1034
Train scores: [0.4496407558383838, 0.5813405079260353, 0.27083309185799814, 0.4025587229806919, 0.23666669695030162] 0.388
Val scores: [0.3499334314596601, 0.6220702166834208, 0.36459265651589584, 0.47294527470048675, 0.5709054294592605]   0.476

Subject: 1038
Train scores: [0.15594304625158115, 0.36843702752512025, 0.24718242742670457, 0.33583171100140585, 0.12065655207631944]       0.246
Val scores: [0.3407679948483705, 0.3576507222864357, 0.30621427011431346, 0.38246936643387036, 0.3092964477227775]   0.339

Subject: 1043
Train scores: [0.21267763034543255, 0.20376125054511962, 0.49989611321285315, 0.20927445609503142, 0.3438550073454868]        0.294
Val scores: [0.2833560267761497, 0.15089000426527466, 0.236647327274029, 0.03939865135677193, 0.0465671093550616]    0.151

Subject: 1048
Train scores: [0.1762437422917564, 0.25528039997621843, 0.17818432863058467, 0.30030656098323844, 0.278115152855797] 0.238
Val scores: [0.44901483302689205, 0.37260218773965903, 0.4956638546081247, 0.6124539718071238, 0.5525132670471014]   0.496

Subject: 1049
Train scores: [0.42066647230315946, 0.1992899990980562, 0.31666904226171394, 0.4617456221544147, 0.3924526944172338] 0.358
Val scores: [0.3659821265981803, 0.3705490732821477, 0.38599084493176716, 0.3622965439038025, 0.43114047179936377]   0.383

Final train: 0.330
Final val: 0.417
'''

'''
Subject: 1004
Train scores: [0.2879885130551797, 0.5783659054375557, 0.7630559390065419, 0.6809010714767753, 0.7802216692108601]      0.618
Val scores: [0.7698484047992291, 0.8729290295971952, 0.3307964476786733, 0.6633215866135038, 0.6244483659119782]        0.652

Subject: 1006
Train scores: [0.1697321084261067, 0.3273698882817782, 0.1671513362346999, 0.6985511576364908, 0.1777038652831778]      0.308
Val scores: [0.40112065078289033, 0.2718381018099528, 0.27054942668225906, 0.527661828081346, 0.4475644208847949]       0.384

Subject: 1007
Train scores: [0.22099974763369185, 0.28945963965827365, 0.49057865928853017, 0.22236674511038085, 0.10358308051720747]         0.265
Val scores: [0.48011766736674666, 0.3698835256186927, 0.5238981078525959, 0.380129265590255, 0.3596283901404491]        0.423

Subject: 1019
Train scores: [0.1395385409065635, 0.28193107676970874, 0.3012700382271679, 0.7053793289929337, 0.3423284689768299]     0.354
Val scores: [0.4789950557594927, 0.4580513419021823, 0.3613252794308348, 0.3881032224740557, 0.3632886127012944]        0.410

Subject: 1020
Train scores: [0.21709629468444466, 0.30275454615685665, 0.10020585156034817, 0.15057584197232185, 0.3303023016779225]  0.220
Val scores: [0.33329795218258956, 0.3632157532005736, 0.32934395039415715, 0.44467241236719157, 0.3735964052806948]     0.369

Subject: 1023
Train scores: [0.15359226101811116, 0.4455176354363766, 0.32071529236067126, 0.1881437770017676, 0.2017768312614409]    0.262
Val scores: [0.5887068345106177, 0.6982225748788926, 0.433628630738661, 0.4266844278261013, 0.458328801551725]  0.521

Subject: 1032
Train scores: [0.12950312369727415, 0.12891216313704082, 0.5776317915460424, 0.17034722467846744, 0.16378076594856883]  0.234
Val scores: [0.4945795608447334, 0.5123778023067321, 0.5222396565333699, 0.5139436374848624, 0.5680491159153062]        0.522

Subject: 1034
Train scores: [0.3983505752588052, 0.3024274315960421, 0.26023463881063247, 0.19153736450548015, 0.3666620959553349]    0.304
Val scores: [0.42828152990210266, 0.5934560449644439, 0.5088777080492881, 0.5385743294211695, 0.5694871966536252]       0.528

Subject: 1038
Train scores: [0.2941587397792426, 0.15923929837571968, 0.19656316335721327, 0.14892337653222612, 0.07869376861166781]  0.176
Val scores: [0.3773657533130761, 0.33948528110941295, 0.2835258727018925, 0.3571975888445153, 0.3105204577268707]       0.334

Subject: 1043
Train scores: [0.17081088411911713, 0.223518600617395, 0.48419310630801543, 0.25231515763749124, 0.43238595332608243]   0.313
Val scores: [0.1534942260174228, 0.14587083757185612, 0.23947221141624842, 0.034946383145269516, 0.06596859647835993]   0.128

Subject: 1048
Train scores: [0.20946366914026487, 0.20949433496500403, 0.18130338761107095, 0.28971098135956463, 0.4898983078755479]  0.276
Val scores: [0.4967152464659982, 0.4362398288765314, 0.42321041464975323, 0.49421189288032513, 0.6383760099445972]      0.498

Subject: 1049
Train scores: [0.33684373200194984, 0.2696324017241575, 0.47680599332934587, 0.3346216568121152, 0.4183296194974395]    0.367
Val scores: [0.4546596420248602, 0.39001929052034956, 0.31695908651558724, 0.41940289741428477, 0.48974442030068344]    0.414

Final train: 0.308
Final val: 0.432
'''

'''
DYSKINESIA
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


