import csv
import matplotlib
import matplotlib.pyplot as plt
# %matplotlib inline
import re

import numpy as np
from scipy import signal
from time import time

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import xgboost as xgb

# Read log file and do some preprocessing
# input: path - file name
# output: datalst - list of data
#         starts - list of line numbers on all occurance of "StartUp!"
def read_log(path):
    print("    Reading file " + path)
    # Read file
    file = open('data/'+path, encoding='windows-1252')

    # Pre process, drop invalid outputs
    starts = []
    file.seek(0)
    data = file.read()
    datalst = data.split('\n')[2:]

    pre_len = len(datalst)
    i, idx = 0, 0
    while i < len(datalst):
        line = datalst[i]
        if re.match(".*Start", line):
            starts.append(idx)
        #if line.count(',') < 6:
        if line.count(',') != 6:
            datalst.remove(line)
            i -= 1
        i, idx = i + 1, idx + 1
    post_len = len(datalst)

    print("        %d invalid data removed." %(pre_len - post_len))
    print("        Lines contain 'StartUp!': ", starts)
    print('        Number of movement samples:', len(starts))
    
    return datalst, starts


# Separate data into 6 channels
# input: sample - list of data
# output: channels - list of length 6, each being data for a channel
def extract_channel(sample):
    channels = [[], [], [], [], [], []]
    for line in sample:
        line_data = line.split(',')
        for i in range(6):
            channels[i].append(line_data[i])
    return list(np.array(channels, dtype=int))


# input: Nxk array (N: #sample points; k: #old channels)
# output: Nx(k+n) array 
def append_channel(channels, n_new_ch):
    ret = list(channels.T)
    ret.extend([list(np.zeros(ret[0].shape))]*n_new_ch)
    ret = np.array(ret).T
    return ret


# input: feature - Nx1 array, prelabeled version
#        bound - int, lower bound of the onset interval length
# output: feature - Nx1 array, relabeled version 
def unlabel_short_onset(feature, bound):
    start_idx, end_idx = 0, 1
    N = len(feature)
    
    while start_idx < len(feature):
        while start_idx < N and feature[start_idx] == 0:
            start_idx += 1
        if start_idx >= N:
            break
        end_idx = start_idx+1
        while end_idx < N and feature[end_idx] > 0:
            end_idx += 1
        count = end_idx - start_idx
        if count < bound:
            feature[start_idx:end_idx] = 0
        start_idx = end_idx
        
    return feature


# For Sample Windows Relabeling
# input: array, last channel for label
# output: a list of sliding windows
def extract_labeled_windows(channels, n_win, n_step, n_win_decision):
    N = channels.shape[0]
    labeled_windows = []
    start_idx, end_idx = 0, n_win
    while end_idx <= N:
        segment = np.copy(channels[start_idx:end_idx, :]) # remember to make a copy
        
        # for the interval of interests (decision window)
        # if more than half of the window is in the onset interval, then label the whole window as onset
        if np.sum(segment[-n_win_decision:,-1] > 0) < n_win_decision/2: 
            segment[-n_win_decision:,-1] = 0 # label as rest
        else: 
            segment[-n_win_decision:,-1] = max(segment[-n_win_decision:,-1]) # label by the movement label
        #segment[:,-1] = (np.sum(segment[:,-1] > 0) >= segment.shape[0]/2) * max(segment[:,-1]) # a slower version
    
        labeled_windows.append(segment)
        start_idx, end_idx = start_idx+n_step, end_idx+n_step
    return labeled_windows


def plot_original_samples(fs, movement_names, files, all_datalst, all_starts):
    all_samples = [[], [], []]

    for label in range(len(movement_names)):
        sample_indicator = 1
        for file_idx in range(len(files[label])): 
            datalst, starts = all_datalst[label][file_idx], all_starts[label][file_idx]
            # extract individual samples
            for i in range(1,len(starts)-2): # discard the first and last sample in each file
                start_idx, end_idx = starts[i], starts[i+1]
                if end_idx-start_idx < 10*fs: # discard the movement samples shorter than 10s
                    continue 
                sample = datalst[start_idx+1*fs:end_idx-1*fs] # discard the first 1s and last 1s of each sample (disturbance from the startup buttons)
                channels = extract_channel(sample)
                channels[0] = [1000 if sample_indicator > 0 else 0 for k in range(len(channels[1]))]
                all_samples[label].extend(list(np.array(channels).T))
                sample_indicator *= -1

    # visualization
    figsize = (40,5)
    # print(np.array(all_samples[0])[:,1])
    for label in range(len(movement_names)):
        plt.figure(figsize=figsize)
        plt.plot(np.array(all_samples[label]))
        plt.title(str(movement_names[label]+" Signals (Original)"))
        plt.xlabel('n (sample point index)')
        plt.ylabel('Intensity')
        plt.show(block=False)






def pre_training(fs, t_win, t_step, t_win_decision, files, movement_names=['Rest', 'Fist', 'Open'], plot_it=[True, True, True], train_test_in_diff_file=False, use_reference=False, tree_depth=12, test_size=0.95):
    
    # read files
    print(f'>>> Now reading training data files...')
    all_datalst, all_starts = [[],[],[]], [[],[],[]]
    for label in range(len(movement_names)):
        for file_idx in range(len(files[label])): 
            datalst, starts = read_log(files[label][file_idx])
            all_datalst[label].append(datalst)
            all_starts[label].append(starts)


    # labeling
    time_start = time()
    n_win, n_step = int(t_win*fs), int(t_step*fs)
    n_win_decision = int(fs*t_win_decision)

    print(f'\n>>> Now labeling the training data...')
    filtered_samples = [[], [], []]
    all_labeled_windows = [] # this is our training dataset
    b, a = signal.butter(2, [8, 15], 'bp', fs=fs, output='ba') # seems to be a nice choice
    d, c = signal.butter(2, 0.2, 'lp', fs=fs, output='ba') # for the feature extraction steps
    for label in range(len(movement_names)):
        print(f'    Processing the {movement_names[label]} signals...')
        sample_indicator = 1
        for file_idx in range(len(files[label])): 
            print(f'        File name: {files[label][file_idx]}')
            datalst, starts = all_datalst[label][file_idx], all_starts[label][file_idx]
            # extract individual samples
            for i in range(0,len(starts)-1): # discard the first and last sample in each file
                start_idx, end_idx = starts[i], starts[i+1]
                if end_idx-start_idx < 10*fs: # discard the movement samples shorter than 10s
                    continue 
                sample = datalst[start_idx+1*fs:end_idx-1*fs] # discard the first 1s and last 1s of each sample (disturbance from the startup buttons)
                channels = extract_channel(sample)
                channels = np.array(channels, dtype=float).T
                
                train_channels = append_channel(channels, 3)[int(0.2*fs):-int(0.2*fs),:] # add 3 new channels for features used in training dataset
                # train_channels = append_channel(channels, 1)[int(0.2*fs):-int(0.2*fs),:] # add 1 new channel for features used in training dataset
                channels = append_channel(channels, 3) # add 3 new channels for features used in onset detection

                """The below chunk is for relabeling of movement interval"""
                # try filtering for each sample
                channels = signal.lfilter(b, a, channels, axis=0)[int(0.2*fs):-int(0.2*fs),:] # discard the glitched intervals
                # add one channel (the 7th) for sample indicator
                channels[:,6] = np.array([1000 if sample_indicator > 0 else 0 for k in range(len(channels[:,0]))])
                # add one channel (the 8th) for extracted feature
                feature = np.mean(channels[:,1:6]**2, axis=1)
                feature = signal.lfilter(d, c, feature, axis=0)
                mean_all = 0.6*np.mean(feature[-int(1.5*fs):]) + 0.5*np.mean(feature[:int(1.5*fs)])
                std_all = np.std(feature[:])
                mean_ref = 0.6*np.mean(feature[-int(1.5*fs):]) + 0.5*np.mean(feature[:int(1.5*fs)])
                std_ref = 0.6*np.std(feature[-int(1.5*fs):]) + 0.5*np.std(feature[:int(1.5*fs)]) # this metrics is slightly different than the one used in onset detection (for practical reasons)
                feature -= mean_all           # normalize the sample feature
                feature /= std_all            # normalize the sample feature
                feature *= 200 # scaling for easier visualization
                channels[:,7] = feature
                # add another channel (the 9th) for the final feature
                threshold = 150
                movement = (feature > threshold) * label
                movement = unlabel_short_onset(movement, int(fs*2)) # finally, unlabel the onset intervals that are shorter than 2 sec
                channels[:,8] = movement # final label

                """The below chunk is for preparing the training dataset"""
                # # add another two channels (the 10th & 11th) for the reference feature we used in feature extraction
                train_channels[:,6] = mean_ref
                train_channels[:,7] = std_ref
                train_channels[:,8] = movement # final label
                # train_channels[:,6] = movement # final label
                # turn into sliding windows
                labeled_windows = extract_labeled_windows(train_channels, n_win, n_step, n_win_decision)
                
                filtered_samples[label].extend(list(channels))
                all_labeled_windows.extend(labeled_windows)
                sample_indicator *= -1
                
    time_end = time()
    print(f'Done! Onset detection and movement labeling finished in {time_end - time_start :.2f} sec.')

    if plot_it[1]:
        # visualize the sample-wise filtered samples
        print(f'\n>>> Now plotting the filtered signals, the onset feature and the movement labels...')
        plt.figure(figsize=(20,3*len(movement_names)))
        for label in range(len(movement_names)):
            plt.subplot(len(movement_names),1,label+1)
            filtered = np.array(filtered_samples[label])

            plt.plot(filtered[:,1:6]) # 5 active channels 
            # plt.plot(filtered[:,6]*0.50) # sample indicator
            plt.plot(filtered[:,7]/2, 'blue') # sample feature
            plt.plot(filtered[:,8]*200, 'black') # movement label
            
            plt.title(str(movement_names[label]+" Signals (Sample-wise Bandpass Filtered)"))
            plt.ylabel('Intensity')
            plt.ylim(-400, 500)
            # plt.draw()
        plt.xlabel('n (sample point index)')
        plt.show(block=False)
    

    # seperate channels
    time_start = time()
    print(f'\n>>> Now extracting the windowed segments for training data...')
    all_labeled_windows = np.array(all_labeled_windows)
    # data_x = all_labeled_windows[:,:,:-1]
    data_x = all_labeled_windows[:,:,:-3]
    data_ref = all_labeled_windows[:,0,-3]
    data_std = all_labeled_windows[:,0,-2]
    time_end = time()
    print(f'Done! Training data extraction finished in {time_end - time_start :.2f} sec.')   



    """                              final feature extraction                              """
    # final feature extraction
    time_start = time()
    print(f'\n>>> Now computing the feature for training data...')
    # step 1
    time_now = time()
    data_feature = signal.lfilter(b, a, data_x, axis=1)**2
    print(f'    Step 1 finished (square of the Butterworth bandpass filtered signal), time used: {time() - time_now:.2f} sec.')
    # step 2
    time_now = time()
    f, e = signal.butter(2, 6, 'lp', fs=400, output='ba') # adapted from b, a
    data_feature = signal.lfilter(f, e, data_feature, axis=1)
    # data_feature_no_ref = data_feature
    print(f'    Step 2 finished (smoothing by Butterworth lowpass filtering), time used: {time() - time_now:.2f} sec.')
    if use_reference:
        # step 3
        time_now = time()
        data_feature = data_feature - data_ref[:,None,None]
        print(f'    Step 3 finished (subtract the reference mean), time used: {time() - time_now:.2f} sec.')
        # step 4
        time_now = time()
        data_feature = data_feature / data_std[:,None,None]
        print(f'    Step 4 finished (normalize the reference standard deviation), time used: {time() - time_now:.2f} sec.')
        # data_feature_with_ref = data_feature
    time_end = time()
    print(f'Done! Training dataset feature extraction finished in {time_end - time_start :.2f} sec.')



    """                                      training                                      """
    # training
    time_start = time()
    print(f'\n>>> Now training the xgboost classifier...')
    X = np.mean(data_feature[:,-n_win_decision:,:], axis=1)
    Y = all_labeled_windows[:,-int(n_win_decision/2),-1]
    plt.figure()
    plt.plot(Y)
    plt.draw()
    if not train_test_in_diff_file:
        train_x, test_x, train_y, test_y = train_test_split(X,Y,test_size=test_size)
    else: 
        train_x, test_x, train_y, test_y = [], [], [], []
        trn_size, all_size = 10, 13
        L = X.shape[0]
        train_x.extend(list(X[int(L*(0+0*trn_size/all_size)):int(L*(0+1*trn_size/all_size))-300]))
        train_x.extend(list(X[int(L*(1+0*trn_size/all_size)):int(L*(1+1*trn_size/all_size))-300]))
        train_x.extend(list(X[int(L*(2+0*trn_size/all_size)):int(L*(2+1*trn_size/all_size))-300]))
        train_y.extend(list(Y[int(L*(0+0*trn_size/all_size)):int(L*(0+1*trn_size/all_size))-300]))
        train_y.extend(list(Y[int(L*(1+0*trn_size/all_size)):int(L*(1+1*trn_size/all_size))-300]))
        train_y.extend(list(Y[int(L*(2+0*trn_size/all_size)):int(L*(2+1*trn_size/all_size))-300]))
        test_x.extend(list(X[300+int(L*(0+1*trn_size/all_size)):int(L*(1+0*trn_size/all_size))]))
        test_x.extend(list(X[300+int(L*(1+1*trn_size/all_size)):int(L*(2+0*trn_size/all_size))]))
        test_x.extend(list(X[300+int(L*(2+1*trn_size/all_size)):int(L*(3+0*trn_size/all_size))]))
        test_y.extend(list(Y[300+int(L*(0+1*trn_size/all_size)):int(L*(1+0*trn_size/all_size))]))
        test_y.extend(list(Y[300+int(L*(1+1*trn_size/all_size)):int(L*(2+0*trn_size/all_size))]))
        test_y.extend(list(Y[300+int(L*(2+1*trn_size/all_size)):int(L*(3+0*trn_size/all_size))]))
        train_x, test_x, train_y, test_y = np.array(train_x), np.array(test_x), np.array(train_y), np.array(test_y)
    print(train_x.shape, test_x.shape, train_y.shape, test_y.shape)
    # print(f'train_y: {train_y};\ntest_y: {test_y}')
    xg_train=xgb.DMatrix(train_x,label=train_y)
    xg_test=xgb.DMatrix(test_x,label=test_y)

    param = {}
    param['objective'] ='multi:softmax'
    # param['eval_metric'] ='merror'
    param['eval_metric'] ='mlogloss'
    param['eta']=0.1
    param['max_depth']=tree_depth
    # param['silent']=1
    # param['nthread']=4
    param['num_class']=3
    watchlist = [(xg_train, 'train'), (xg_test, 'test')]
    num_round=5
    bst = xgb.train(param, xg_train, num_round, watchlist)

    time_end = time()
    print(f'Done! Training finished in {time_end - time_start :.2f} sec.')
    print(f'Tree parameters: {param}')

    # prediction
    time_start = time()
    print(f'\n>>> Now evualuating the xgboost classifier...')
    pred = bst.predict(xg_test)
    time_end = time()
    print(f'Done! Prediction finished in {time_end - time_start :.5f} sec.')

    # classifier evaluation
    accuracy = accuracy_score(test_y, pred)
    conf_mat = confusion_matrix(test_y, pred, labels=[0,1,2], normalize='true')
    print(f'Accuracy: {accuracy * 100:.2f}%') 
    print(f'Confusion Matrix (normalized over the true condition):\n {conf_mat}')
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=None)
    if plot_it[2]:
        disp.plot() 

    plt.figure()
    plt.plot(pred)
    plt.plot()
    plt.draw()

    plt.show()

    return bst


""" Reference no longer needed
# input:    channels - 
# 
# 
def extract_reference(channels):
    b, a = signal.butter(2, [8, 15], 'bp', fs=fs, output='ba') # seems to be a nice choice
    d, c = signal.butter(2, 0.2, 'lp', fs=fs, output='ba') # for the feature extraction steps
    # filtering the sample
    channels = signal.lfilter(b, a, channels, axis=0)[int(0.2*fs):-int(0.2*fs),:] # discard the glitched intervals
    # add one channel (the 7th) for sample indicator
    channels[:,6] = np.array([1000 if sample_indicator > 0 else 0 for k in range(len(channels[:,0]))])
    # add one channel (the 8th) for extracted feature
    feature = np.mean(channels[:,1:6]**2, axis=1)
    feature = signal.lfilter(d, c, feature, axis=0)
    mean_ref = 0.6*np.mean(feature[-int(1.5*fs):]) + 0.5*np.mean(feature[:int(1.5*fs)])
    std_ref = 0.6*np.std(feature[-int(1.5*fs):]) + 0.5*np.std(feature[:int(1.5*fs)])

    return mean_ref, std_ref
"""


""" Reference no longer needed
# inupt:    bst - the xgboost classifier
#           data_win_x - (n_win, n_ch) array, a 5-sec data window of the input raw signal
#           mean_ref - (n_win,) array
#           std_ref - (n_win,) array
def predict(fs, t_win_decision, win_x, mean_ref, std_ref, bst):
"""
# inupt:    bst - the xgboost classifier
#           data_win_x - (n_win, n_ch) array, a 5-sec data window of the input raw signal
def predict(fs, t_win_decision, data_win_x, bst):
    n_win_decision = int(t_win_decision*fs)

    # feature extraction
    b, a = signal.butter(2, [8, 15], 'bp', fs=fs, output='ba')
    f, e = signal.butter(2, 6, 'lp', fs=fs, output='ba') # adapted from b, a
    data_feature = signal.lfilter(b, a, data_win_x, axis=0)**2 # notice the axis=0 is different from pre-training (here only a single window is passed)
    data_feature = signal.lfilter(f, e, data_feature, axis=0) # notice the axis=0 is different from pre-training
    # data_feature = (signal.lfilter(f, e, signal.lfilter(b, a, win_x, axis=0)**2, axis=0) - mean_ref[:,None]) / std_ref[:,None]
    
    X = np.mean(data_feature[None,-n_win_decision:,:], axis=1) # extract the feature in the interested interval (e.g., the last 0.5 sec of the window)
    xg_x = xgb.DMatrix(X)
    pred = bst.predict(xg_x)
    return pred



def main():
    fs = 500
    t_win, t_step = 5.000, 0.050
    t_win_decision = 0.500 

    # files = [['rest1.log', 'rest2.log', 'rest3.log', 'rest4.log', 'rest5.log', 'rest6.log', 'rest7.log', 'rest8.log', 'rest9.log', 'rest10.log'],  
    #          ['fist1.log', 'fist2.log', 'fist3.log', 'fist4.log', 'fist5.log', 'fist6.log', 'fist7.log', 'fist8.log', 'fist9.log', 'fist10.log'], 
    #          ['open1.log', 'open2.log', 'open3.log', 'open4.log', 'open5.log', 'open6.log', 'open7.log', 'open8.log', 'open9.log', 'open10.log']]
    files = [['rest1.log', 'rest3.log', 'rest5.log', 'rest7.log'],
             ['fist1.log', 'fist3.log', 'fist5.log', 'fist7.log'],
             ['open1.log', 'open3.log', 'open5.log', 'open7.log']] 
    # files = [['rest1.log', 'rest5.log'],  
    #          ['fist1.log', 'fist5.log'], 
    #          ['open1.log', 'open5.log']]
    # files = [['rest_new.log', "rest_test.log"],  
    #          ['fist_new.log', "fist_test.log"], 
    #          ['open_new.log', "open_test.log"]] 
    # files = [["rest_test.log"],  
    #          ["fist_test.log"], 
    #          ["open_test.log"]] 
    movement_names = ['Rest', 'Fist', 'Open']

    bst = pre_training(fs, t_win, t_step, t_win_decision, files, movement_names, plot_it=[False, True, True], 
                        train_test_in_diff_file=False, use_reference=False, tree_depth=6, test_size=0.2)
    bst.save_model("model_xgboost.json")

    # unit testing
    data_win_x = np.ones((int(t_win*fs),6))
    pred = predict(fs, t_win_decision, data_win_x, bst)
    print(f'\n\n>>> Unit Test of predict(): pred={pred}')

if __name__ == "__main__":
    main()