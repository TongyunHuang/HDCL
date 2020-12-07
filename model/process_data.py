'''
Package Import
'''
# Computation & Signal Processing
# from scipy import signal
import numpy as np
import pandas as pd
# import pylab as pl
import pickle
import scipy.io as spio
import matplotlib.pyplot as plt
# from biosppy import storage #biosppy package for ecg signal analysis
from biosppy.signals import ecg
import time 
import math
from sklearn import preprocessing
# Self defined package
from utils import * #import data import, clean up and sampling functions. 
from ECG_feature_extractor_1000 import *

'''
class Processor
read data from .csv file 
'''

class Processor:
    '''
    Global variables
    '''
    # initializa variables
    fs_acc = 148.148 #actual 148.15 
    fs_emg = 1925.9258 #from 1/df['X[s]'][1] 
    fs_ecg = 1925.9258
    ## Directory path
    data_dir_fNIRS = 'fNIRS'
    data_dir = '../csv'
    file_dict = {'302':{'nw1':'/OA_2019_302_NW1_Rep_1.2.csv',
                        'nw2':'/OA_2019_302_NW2_Rep_1.5.csv',
                        'p1':'/OA_2019_302_P1_Rep_1.3.csv',
                        'p2':'/OA_2019_302_P2_Rep_1.4.csv'}}
    treadMill_file = {'302':{'nw1':'/OA_302_TM_N1.csv',
                             'nw2':'/OA_302_TM_N2.csv',
                             'p1' :'/OA_302_TM_P1.csv',
                             'p2' :'/OA_302_TM_P2.csv'}}

    '''
    Class constructor, can be extend by allowing parameter
    @ param file_name: String of the filename, for specific person one trial
    '''
    def __init__(self, participant, trial):
        ## Existed string definition in Abdul file
        self.trial = trial
        self.data_dir_treadmill = '../Treadmill data_trial1-4'
        self.file3_rawdata = self.treadMill_file[participant][trial]
        ## Redefine "hardcode" file name
        self.trialFile = self.file_dict[participant][trial]
        ## Other variables initialization
        self.interval_index_span = 0.0
        self.lf_hf_store = []
        self.sdnn_store = []
        self.file_data = {}
        self.treadmill_data = {}
        self.matlabEMG = {}
        self.intervals = []
        self.process_data_file()
        

    def process_data_file(self):
        # fs: sampling frequency
        # df: reading a csv file and store it in panads dataframe (http://pandas.pydata.org/pandas-docs/stable/)
        # loading datafile 
        load_file = load_data( self.trialFile , self.data_dir )
        # data clean up and assignment
        self.file_data = delsys_cleanup(load_file)
        ## Other file processing
        self.matlabEMG = spio.loadmat('../302_p1_EMG_datamatlab.mat', squeeze_me=True)
        # if(self.trial != 'p1'):
        # self.treadmill_data = load_data(self.file3_rawdata, self.data_dir_treadmill,1)
        # else:
        self.treadmill_data = load_data(self.file3_rawdata, self.data_dir_treadmill)
        ## Array with interval information
        intervals = self.setInterval(30, 5, self.fs_ecg, self.file_data.shape[0])
        ## Extract different info such as lfhf and sdnn
        pack = []
        ecg_out = []
        RR_store = []
        for idx, val in enumerate(intervals[:-1]):
            current_segment = self.file_data[val[0]:val[1]].ecg
            fs = self.fs_ecg
            BS_signal_analysis = ecg.ecg(signal=current_segment, sampling_rate=fs, show=False)
            RR = BS_signal_analysis['rpeaks']
            RR_store.append(RR)
            A, B = freq_ratio_hybrid(current_segment, fs=fs, RR=BS_signal_analysis['rpeaks'], method = 'periodogram', factor = 1)
            pack.append(A)
            ecg_out.append(B)
        self.setLfhf(pack)
        self.setSdnn(RR_store)
        
    '''
    Set the interval we want to extract data
    @ param interval_length: The length of interval
            stride: stride of interval
    '''
    def setInterval(self, interval_length, stride, frequency, time_length):
        # interval_index_span = self.cohort['302']['p1'].fs * interval_length 
        stride_length = np.int( stride * frequency)
        interval_length = np.int(interval_length * frequency)
        total_interval_over_strides = np.int(time_length // stride_length)
        print(total_interval_over_strides)
        # Array with interval information
        self.intervals = [None]*(total_interval_over_strides)
        for i in range(total_interval_over_strides):
            self.intervals[i] = [i * stride_length, i * stride_length + interval_length]
        return self.intervals     
        
    '''
    Extract Low frequency/ High frequency data from pack
    '''
    def setLfhf(self, pack):
        self.lf_hf_store = []
        for entry in pack:
            self.lf_hf_store.append(entry['lf_hf'])
        # self.lf_hf_store = preprocessing.scale(self.lf_hf_store)
            # self.lf_hf_store.append(math.log(entry['lf_hf']))
        
    '''
    Extract sdnn data from RR_store, normalize SDNN data to avoid bias
    '''
    def setSdnn(self, RR_store):
        self.sdnn_store = []
        for entry in RR_store:
            # ibi = np.diff(entry)
            sdnn = np.std(entry)
            self.sdnn_store.append(sdnn/400)
        # self.sdnn_store = preprocessing.scale(self.sdnn_store)
            # self.sdnn_store.append(math.log(sdnn))
        
    '''
    Get LF/HF data for current file
    '''
    def getLfhf(self):
        return self.lf_hf_store

    '''
    Get sdnn data for current file
    '''
    def getSdnn(self):
        list_ = []
        for i in self.sdnn_store:
            list_.append(i)
        return self.sdnn_store

    '''
    2D array with x-axis: sdnn, y-axis: lf/hf
    Useful for plotting
    '''
    def get_sdnn_lfhf_array(self):
        X_train = [self.sdnn_store,self.lf_hf_store]
        preprocessing.StandardScaler().fit(X_train)
        length = len(self.lf_hf_store)
        X = np.zeros(shape=(length,2))
        for i in range( length ):
            X[i] = ( self.sdnn_store[i] , self.lf_hf_store[i] )
        return X
    
    '''
    return a diction (key, value)
    key: (sdnn, lf/hf) value: index with the (sdnn,lf/hf) data pair
    '''
    def get3d(self):
        dict_ = {}
        length = len(self.lf_hf_store)
        # Y = np.zeros(shape=(length,3))
        for i in range( length ):
            # if i > 35:
            #     break
            dict_[(self.sdnn_store[i], self.lf_hf_store[i])] = (self.trial,i)
            
        return dict_
        
    '''
    Get treadMill speed data
    '''
    def getTreadMData(self):
        # self.treadmill_data = load_data(self.file3_rawdata, self.data_dir_treadmill)
        return self.treadmill_data

    '''
    Get EMG data
    '''
    def getEMG(self):
        return self.matlabEMG
    
    def getInterval(self):
        return self.intervals

    # def load_tdata(file_name,data_dir='csv', gdrive=True):
    #     file = data_dir*gdrive +  file_name
    #     df = pd.read_csv(file)
    # return df

def main():
    p1Data = Processor('302','nw1')
    treadmill_data = p1Data.getTreadMData()
    print(treadmill_data.Time)
    plt.plot(treadmill_data.Time, treadmill_data.Speed)
    plt.show()
    # processor = Processor('302','nw1')
    

if __name__=="__main__": 
    main()
    
