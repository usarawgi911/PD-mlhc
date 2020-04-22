import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import os, math, glob

# import tensorflow as tf
# from tensorflow.keras.layers import *
# from tensorflow.keras.models import Model

def prepare_data(dataset_dir='../CIS'):
	# data_files = sorted(glob.glob(os.path.join(dataset_dir, 'training_data/*.csv')))
	label_file = glob.glob(os.path.join(dataset_dir, 'data_labels/CIS-PD_Training_Data_IDs_Labels.csv'))[0]
	labels_df = pd.read_csv(label_file)
	# print(labels_df.shape)
	# print(labels_df.head())

	subject_ids, counts = np.unique(labels_df['subject_id'], return_counts=True)
	# print(subject_ids, counts)

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
		# print()
		# print(subject_id)
		for measurement_id in measurement_ids[idx]:
			file = glob.glob(os.path.join(dataset_dir, 'training_data/{}.csv'.format(measurement_id)))[0]
			subject_files.append(file)

			data = np.delete(np.array(pd.read_csv(file)), 0, 1)
			subject_data.append(data)

			n_data = data.shape[0]
			subject_n_data.append(n_data)

		# print(np.min(subject_n_data), np.mean(subject_n_data), np.max(subject_n_data))
		all_data.append(subject_data)
		all_n_data.append(subject_n_data)

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

def cleaned_data(all_data, labels):

	X, y = [], [] # X ~ (subjects, datapoints, timesteps, 3)
	for idx, subject_label in enumerate(labels):
		# X_subject is a list of np arrays shape (timesteps, 3)
		X_subject, y_subject = [], []
		for jdx, label in enumerate(subject_label):
			if not math.isnan(label):
				y_subject.append(np.float32(label))
				X_subject.append(all_data[idx][jdx].astype(np.float32))
		if len(X_subject)!=0:	X.append(X_subject)
		if len(y_subject)!=0:	y.append(y_subject)

	return X, y

# def create_model(input_shape):
# 	x = Input(shape=input_shape)
# 	x = Conv1D(16, 3, )

# baseline()
_, _, all_data, _, on_off_labels, dyskinesia_labels, tremor_labels = prepare_data()
X_on_off, y_on_off = cleaned_data(all_data, on_off_labels)
print(len(X_on_off), len(y_on_off))
print(len(X_on_off[0]), len(y_on_off[0]))
print(X_on_off[0][0].shape, y_on_off[0][0])




