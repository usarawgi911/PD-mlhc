import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm
from fastdtw import fastdtw
import pickle

# find the distance between a set of samples
# it take too long to run this on every sample (and it's not very insightful)

sampling_rate = 0.1

base = '/Users/daphne/Dropbox (MIT)/'

subjects = [1004, 1006, 1007, 1019, 1020, 1023, 1032, 1034, 1038, 1043, 1048, 1049]

# load in labels

path = base + "/pd-mlhc/CIS/data_labels/CIS-PD_Training_Data_IDs_Labels.csv"
 
labels = pd.read_csv(path)

# load the waveforms, put everything into lists
print('\nloading waveforms...\n')


wfs = []
tremor = []
for s in tqdm(subjects) :
    labels_per = list(labels[labels['subject_id'] == s]['measurement_id'])
    tremor.append(list(labels[labels['subject_id'] == s]['tremor']))
    wfs_per = []
    for loc in labels_per :
        path = base + "pd-mlhc/CIS/training_data/" + loc + '.csv'
        wf = pd.read_csv(path)
        wfs_per.append(np.asarray(wf.loc[:,['X','Y','Z']]))
    wfs.append(wfs_per)
    

tremor_all = []
pid = []
wfs_all = []

i = 0
for w in range(len(wfs)) :
    for wf in range(len(wfs[w])):
        r = np.random.rand()
        if r < sampling_rate :
            wfs_all.append(wfs[w][wf])
            pid.append(i)
            tremor_all.append(tremor[w][wf])
        else :
            continue
    i += 1

print('\ncomputing distances\n')

dist_array = np.zeros((len(wfs_all),len(wfs_all)))

for i in tqdm(range(len(wfs_all))) :
    for j in tqdm(range(len(wfs_all))) :
        dist_array[i,j] = fastdtw(wfs_all[i],wfs_all[j])[0]

print('\nsaving variables\n')


pickle.dump( [dist_array,pid,tremor_all,wfs_all], open( "dist_set.p", "wb" ) )