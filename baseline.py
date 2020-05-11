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
os.environ["CUDA_VISIBLE_DEVICES"]="1"

def prepare_data(condition=False):
	'''
	Saves data in .npy files to avoid date parsing delays in every run
	condition = True allows medication status to be used as an input feature for modelling drug response / treatment effect 
	'''
	if condition==False:
		dataset_dir = '../CIS_condition_false'
	else:
		dataset_dir = '../CIS_condition_true'
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

	np.save(os.path.join(dataset_dir, 'subject_ids.npy'), np.array(subject_ids))
	np.save(os.path.join(dataset_dir, 'measurement_ids.npy'), np.array(measurement_ids))
	np.save(os.path.join(dataset_dir, 'all_data.npy'), np.array(all_data))
	np.save(os.path.join(dataset_dir, 'all_n_data.npy'), np.array(all_n_data))
	np.save(os.path.join(dataset_dir, 'on_off_labels.npy'), np.array(on_off_labels))
	np.save(os.path.join(dataset_dir, 'dyskinesia_labels.npy'), np.array(dyskinesia_labels))
	np.save(os.path.join(dataset_dir, 'tremor_labels.npy'), np.array(tremor_labels))

	print('\n..... Data saved\n')

	return subject_ids, measurement_ids, all_data, all_n_data, on_off_labels, dyskinesia_labels, tremor_labels

def load_data(condition=False):
	'''
	Loads data (unpadded) from .npy files 
	'''
	if condition==False:
		dataset_dir = '../CIS_condition_false'
	else:
		dataset_dir = '../CIS_condition_true'
	subject_ids = np.load(os.path.join(dataset_dir, 'subject_ids.npy'), allow_pickle=True)
	measurement_ids = np.load(os.path.join(dataset_dir, 'measurement_ids.npy'), allow_pickle=True)
	all_data = np.load(os.path.join(dataset_dir, 'all_data.npy'), allow_pickle=True)
	all_n_data = np.load(os.path.join(dataset_dir, 'all_n_data.npy'), allow_pickle=True)
	on_off_labels = np.load(os.path.join(dataset_dir, 'on_off_labels.npy'), allow_pickle=True)
	dyskinesia_labels = np.load(os.path.join(dataset_dir, 'dyskinesia_labels.npy'), allow_pickle=True)
	tremor_labels = np.load(os.path.join(dataset_dir, 'tremor_labels.npy'), allow_pickle=True)

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

def extendData(x):

	extended_x = []

	dataLength = max(list(map(lambda i: i.shape[0], x)))
	for x_ in x:

		time = np.array(range(x_.shape[0]))
		xAxis, yAxis, zAxis = x_[:,0], x_[:,1], x_[:,2]

		#Spherical Coordinate Transformation
		rAxis = np.sqrt(xAxis**2 + yAxis**2 + zAxis**2)
		thetaAxis = np.arccos(zAxis/rAxis)
		phiAxis = np.arctan2(yAxis,xAxis)
		
		#Data Cleanup
		epsilon = 0.05 #Error rate
		indexL = list(filter(lambda i: i > 1-epsilon and i < 1+epsilon, rAxis))
		finalL = np.where(rAxis == indexL[-1])[0][0]
		initialL = np.where(rAxis == indexL[0])[0][0]

		timeL = time[initialL:finalL]
		timeL = timeL - timeL[0]
		rAxisL = rAxis[initialL:finalL]
		thetaAxisL = thetaAxis[initialL:finalL]
		phiAxisL = phiAxis[initialL:finalL]
		
		#Extending the data
		while len(timeL)<dataLength:
			
			timeL = np.append(timeL,timeL+timeL[-1]+timeL[1])
			rAxisL = np.append(rAxisL,rAxisL) #Concatenating the absolute value of acceleration

			#Rotation in theta and phi axis
			thetaAxisL = np.append(thetaAxisL,thetaAxisL+thetaAxisL[-1]-thetaAxisL[0])
			phiAxisL = np.append(phiAxisL,phiAxisL+phiAxisL[-1]-phiAxisL[0])
			
		xAxisL = rAxisL*np.sin(thetaAxisL)*np.cos(phiAxisL)
		yAxisL = rAxisL*np.sin(thetaAxisL)*np.sin(phiAxisL)
		zAxisL = rAxisL*np.cos(thetaAxisL)
		
		timeL = timeL[0:dataLength]
		xAxisL = xAxisL[0:dataLength]
		yAxisL = yAxisL[0:dataLength]
		zAxisL = zAxisL[0:dataLength]

		extended_x.append(np.stack((xAxisL, yAxisL, zAxisL), axis=1))
	
	return np.array(extended_x)

def create_model(input_shape):
	'''
	Creates FCN
	'''
	if condition==True:
		input_data = Input(shape=input_shape)

		x = BatchNormalization()(input_data)

		x = Conv1D(16, 7, strides=3, padding='valid')(x)
		x = LeakyReLU()(x)
		x = Conv1D(16, 7, strides=3, padding='valid')(x)
		x = LeakyReLU()(x)
		x = MaxPool1D()(x)
		x = BatchNormalization()(x)

		# x = Conv1D(16, 5, strides=2, padding='valid')(x)
		# x = LeakyReLU()(x)
		# x = Conv1D(16, 5, strides=2, padding='valid')(x)
		# x = LeakyReLU()(x)
		# x = MaxPool1D()(x)
		# x = BatchNormalization()(x)

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

	else:
		input_data = Input(shape=input_shape)
		input_medication = Input(shape=(5))

		x = BatchNormalization()(input_data)

		x = Conv1D(16, 7, strides=3, padding='valid')(x)
		x = LeakyReLU()(x)
		x = Conv1D(16, 7, strides=3, padding='valid')(x)
		x = LeakyReLU()(x)
		x = MaxPool1D()(x)
		x = BatchNormalization()(x)

		# x = Conv1D(16, 5, strides=2, padding='valid')(x)
		# x = LeakyReLU()(x)
		# x = Conv1D(16, 5, strides=2, padding='valid')(x)
		# x = LeakyReLU()(x)
		# x = MaxPool1D()(x)
		# x = BatchNormalization()(x)

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

		x = Concatenate()([x, input_medication])

		x = K.expand_dims(x, axis=1)
		x = BatchNormalization()(x)

		x = Conv1D(1, 1, strides=1, padding='valid')(x)
		x = ReLU(max_value=4)(x)

		model = Model([input_data, input_medication], x)

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

def training(X, Y, medication, folds=5):
	'''
	Performs 5-fold cross-validated training per subject
	Reports 5-fold train and val RMSE scores per subject, as well as final mean and weighted scores 
	'''

	all_train_scores, all_val_scores, all_val_rounded_scores = [], [], []

	for index, x in enumerate(X): # iterating over subjects

		print('\nSubject {} starting\n'.format(str(subject_ids[index])))

		# x = pad(x).astype(np.float32)
		# print(len(x))
		# print(x[0].shape)
		x = extendData(x).astype(np.float32)
		# print(x.shape)
		# exit()
		y = np.array(Y[index]).astype(np.float32)

		p = np.random.permutation(x.shape[0])
		x, y = x[p], y[p]
		med = np.array(medication[index])[p]

		patience = 500 # 200
		fold = 0
		epochs = 1000 # 500
		batch_size = 8 # 16
		learning_rate = 1e-3

		# LR 1e-5 # 0.56, 0.74 (p) 0.58, 0.7 || 0.53 0.7
		# 0.63, 0.49 (no p) # LR (0.5 * 1e-4) 0.73, 0.35

		model_dir = 'models'
		model_dir = 'models-re-re-re' # running in tmux 3 with scalers
		model_dir = 'models-re-re-re-re' # running in tmux 4 w/o scalers
		timeString = time.strftime("%Y%m%d-%H%M%S", time.localtime())
		log_name = '{}'.format(timeString)
		log_name = str(model_dir)
		train_scores, val_scores, val_rounded_scores = [], [], []

		for train_index, val_index in KFold(folds).split(x):

			fold+=1
			x_train, x_val = x[train_index], x[val_index]
			y_train, y_val = y[train_index], y[val_index]
			med_train, med_val = med[train_index], med[val_index]
			med_train, med_val = tf.keras.utils.to_categorical(med_train, num_classes=5), tf.keras.utils.to_categorical(med_val, num_classes=5)

			print('x train shape', x_train.shape)
			print('x val shape', x_val.shape)

			# scalers = {}
			# for i in range(x_train.shape[1]):
			# 	scalers[i] = StandardScaler()
			# 	x_train[:, i, :] = scalers[i].fit_transform(x_train[:, i, :]) 

			# for i in range(x_val.shape[1]):
			# 	x_val[:, i, :] = scalers[i].transform(x_val[:, i, :])
			
			model = create_model(input_shape=x_train.shape[1:])

			model.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
			 loss=tf.keras.losses.mean_squared_error)

			checkpointer = tf.keras.callbacks.ModelCheckpoint(
							os.path.join(model_dir, str(subject_ids[index]), '{}.h5'.format(fold)), monitor='val_loss', verbose=0, save_best_only=True,
							save_weights_only=False, mode='auto', save_freq='epoch')

			tensorboard = tf.keras.callbacks.TensorBoard(log_dir='logs_new/{}@{}@{}'.format(str(log_name), str(subject_ids[index]), str(fold)), histogram_freq=1, write_graph=True, write_images=False)
			es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.001, patience=patience, restore_best_weights=True, verbose=0, mode='auto')			
			
			if condition==True:
				model.fit(x_train, y_train,
								batch_size=batch_size,
								epochs=epochs,
								verbose=1,
								callbacks=[checkpointer, tensorboard, es],
								validation_data=(x_val, y_val))
			else:
				model.fit([x_train, med_train], y_train,
								batch_size=batch_size,
								epochs=epochs,
								verbose=1,
								callbacks=[checkpointer, tensorboard, es],
								validation_data=([x_val, med_val], y_val))				

			es_epoch = es.stopped_epoch
			print(es_epoch)
			if es_epoch==patience:
				raise Exception('Fold {} for subject {} not training'.format(fold, str(subject_ids[index])))

			model = tf.keras.models.load_model(os.path.join(model_dir, str(subject_ids[index]), '{}.h5'.format(fold)))
			
			if condition==True:
				train_preds = model.predict(x_train)
				val_preds = model.predict(x_val)
				train_score = math.sqrt(model.evaluate(x_train, y_train, verbose=0))
				val_score = math.sqrt(model.evaluate(x_val, y_val, verbose=0))
			else:
				train_preds = model.predict([x_train, med_train])
				val_preds = model.predict([x_val, med_val])
				train_score = math.sqrt(model.evaluate([x_train, med_train], y_train, verbose=0))
				val_score = math.sqrt(model.evaluate([x_val, med_val], y_val, verbose=0))

			print()
			print('Preds train \t Preds train rounded \t Y train')
			for j, value in enumerate(y_train):
				print('{:3f} \t {} \t\t\t {}'.format(train_preds[j,0,0], round(train_preds[j,0,0]), y_train[j]))
			print()
			print('Preds val \t Preds val rounded \t Y val')
			for j, value in enumerate(y_val):
				print('{:3f} \t {} \t\t\t {}'.format(val_preds[j,0,0], round(val_preds[j,0,0]), y_val[j]))

			train_scores.append(train_score)
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
	print('~ FINAL RESULTS ~')
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
	print('Final val score: {:.3f}'.format(final_val_score))
	print('Final weighted train score: {:.3f}'.format(final_train_weighted_score))
	print('Final weighted val score: {:.3f}'.format(final_val_weighted_score))
	print('Final root weighted train score: {:.3f}'.format(final_train_root_weighted_score))
	print('Final root weighted val score: {:.3f}'.format(final_val_root_weighted_score))

	print()
	print('Batch size used:', batch_size)
	print('Models in', model_dir)	
	
	return model.summary()


# model = create_model(input_shape=(60000,3))
# print(model.summary())

# baseline()
condition = False
# subject_ids, _, all_data, _, on_off_labels, dyskinesia_labels, tremor_labels = prepare_data(condition=condition)
subject_ids, measurement_ids, all_data, _, on_off_labels, dyskinesia_labels, tremor_labels = load_data(condition=condition)
# new_baseline()
# print(on_off_labels.shape)
# print(len(on_off_labels[0]))
# print(len(all_data[0]))
# print(all_data[0][0].shape)
# exit()

start = time.time()
summary = training(all_data, tremor_labels, on_off_labels)
end = time.time()
time_taken = (end - start)/60 # in minutes
print('\nTraining of all subjects took {:.3f} minutes\n'.format(time_taken))
print(summary)

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


