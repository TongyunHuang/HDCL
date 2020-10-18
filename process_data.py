'''
Package Import
'''
# Computation & Signal Processing
from scipy import signal
import numpy as np
import pandas as pd
import pylab as pl
import pickle
import scipy.io as spio
#biosppy package for ecg signal analysis
from biosppy import storage
from biosppy.signals import ecg
from utils import * #import data import, clean up and sampling functions. 
import time 
from ECG_feature_extractor_1000 import *

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
    data_dir = 'csv'
    file_dict = {'302':{'nw1':'/OA_2019_302_NW1_Rep_1.2.csv',
                        'nw2':'/OA_2019_302_NW2_Rep_1.5.csv',
                        'p1':'/OA_2019_302_P1_Rep_1.3.csv',
                        'p2':'/OA_2019_302_P2_Rep_1.4.csv'}}

    '''
    Class constructor, can be extend by allowing parameter
    @ param file_name: String of the filename, for specific person one trial
    '''
    def __init__(self, participant, trial):
        ## Existed string definition in Abdul file
        self.data_dir_treadmill = 'Treadmill data_trial1-4'
        self.file3_rawdata = '/OA_302_P1_RAWDATA.csv'
        self.file3_fNIRS = '/OA_FNIRS_2019_WALK_306_oxydata.txt'
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
        # loading and parsing data
        load_file =   load_data( self.trialFile , self.data_dir )
        # data clean up and assignment
        self.file_data = delsys_cleanup(load_file)
        ## Other file processing
        self.matlabEMG = spio.loadmat('302_p1_EMG_datamatlab.mat', squeeze_me=True)
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
        
    '''
    Extract sdnn data from RR_store, normalize SDNN data to avoid bias
    '''
    def setSdnn(self, RR_store):
        self.sdnn_store = []
        for entry in RR_store:
            # ibi = np.diff(entry)
            sdnn = np.std(entry)
            self.sdnn_store.append(sdnn/400)
        
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
        length = len(self.lf_hf_store)
        X = np.zeros(shape=(length,2))
        for i in range( length ):
            X[i] = [ self.sdnn_store[i] , self.lf_hf_store[i] ]
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
            dict_[(self.sdnn_store[i], self.lf_hf_store[i])] = i
            
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
'''
Preprocess data for participant 302
'''
def get302Data():
    data_dict = {'302': { 'p1':{} , 'p2':{} , 'nw1':{} , 'nw2':{} } }
    # Data for participant 302, trial p1 
    p1Data = Processor('302','p1')
    data_dict['302']['p1']['data'] = p1Data
    data_dict['302']['p1']['lfhf'] = p1Data.getLfhf()
    data_dict['302']['p1']['sdnn'] = p1Data.getSdnn()
    data_dict['302']['p1']['sd_lf'] = p1Data.get_sdnn_lfhf_array()
    print("Finish getting data for: p1")
    # Data for participant 302, trial p2
    p2Data = Processor('302','p2')
    data_dict['302']['p2']['data'] =  p2Data
    data_dict['302']['p2']['lfhf'] =  p2Data.getLfhf()
    data_dict['302']['p2']['sdnn'] =  p2Data.getSdnn()
    data_dict['302']['p2']['sd_lf'] = p2Data.get_sdnn_lfhf_array()
    print("Finish getting data for: p2")
    # Data for participant 302, trial nw1
    nw1Data = Processor('302','nw1')
    data_dict['302']['nw1']['data'] =  nw1Data
    data_dict['302']['nw1']['lfhf'] =  nw1Data.getLfhf()
    data_dict['302']['nw1']['sdnn'] =  nw1Data.getSdnn()
    data_dict['302']['nw1']['sd_lf'] = nw1Data.get_sdnn_lfhf_array()
    print("Finish getting data for: nw1")
    # Data for participant 302, trial nw2
    nw2Data = Processor('302','nw2')
    data_dict['302']['nw2']['data'] =  nw2Data
    data_dict['302']['nw2']['lfhf'] =  nw2Data.getLfhf()
    data_dict['302']['nw2']['sdnn'] =  nw2Data.getSdnn()
    data_dict['302']['nw2']['sd_lf'] = nw2Data.get_sdnn_lfhf_array()
    print("Finish getting data for: nw2")
    return data_dict
'''
Execute from here, for testing
'''
def main():
    processor = Processor('302','nw1')
    

if __name__=="__main__": 
    main()
    



