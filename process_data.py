# %load_ext autoreload
# %autoreload 2
'''
Package Import
'''
# ML libraries 
# from sklearn.cluster import KMeans
# from sklearn.neighbors.kde import KernelDensity

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



# Plotting
# import matplotlib.pyplot as plt
# import plotly
# import plotly.graph_objs as go
# import plotly.io as pio
# plotly.offline.init_notebook_mode(connected=True)
# %matplotlib widget

from utils import * #import data import, clean up and sampling functions. 
# import time
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

    '''
    Class constructor, can be extend by allowing parameter
    @ param file_name: String of the filename, for specific person one trial
    '''
    def __init__(self, file_name):
        ## Existed string definition in Abdul file
        self.data_dir_treadmill = 'Treadmill data_trial1-4'
        self.file3_clientinfo = 'OA_302_P1_CLIENTINFO'
        self.file3_cueing = 'OA_302_N1_CUEING'
        self.file3_gaitcycles = 'OA_201_N1_GAITCYCLES'
        self.file3_rawdata = '/OA_302_P1_RAWDATA.csv'
        self.file3_fNIRS = '/OA_FNIRS_2019_WALK_306_oxydata.txt'
        ## Redefine "hardcode" file name
        self.df_file = '/OA_2019_302_NW1_Rep_1.2.csv'
        self.df_file1 = '/OA_2019_302_NW2_Rep_1.5.csv'
        self.df_file2 = '/OA_2019_302_P1_Rep_1.3.csv'
        self.df_file3 = '/OA_2019_302_P2_Rep_1.4.csv'
        ## Dictionary structure to store processed data
        self.cohort = {}
        ## Other variables initialization
        self.interval_index_span = 0.0
        self.lf_hf_store = []
        self.sdnn_store = []
        self.treadmill_data = {}
        self.matlabEMG = {}
        self.process_data_file()
        

    def process_data_file(self):
        # fs: sampling frequency
        # df: reading a csv file and store it in panads dataframe (http://pandas.pydata.org/pandas-docs/stable/)
        # gdrive = True
        # loading and parsing data
        df =   load_data( self.df_file , self.data_dir )
        df2 = load_data( self.df_file1 , self.data_dir)
        df3 = load_data( self.df_file2 , self.data_dir)
        df4 = load_data( self.df_file3 , self.data_dir)
        # data clean up and assignment
        nw1 = delsys_cleanup(df)
        nw2 = delsys_cleanup(df2)
        p1 = delsys_cleanup(df3)
        p2 = delsys_cleanup(df4)
        # delsys_cleanup?
        ## Other file processing
        self.matlabEMG = spio.loadmat('302_p1_EMG_datamatlab.mat', squeeze_me=True)
        self.treadmill_data = load_data(self.file3_rawdata, self.data_dir_treadmill)
        # fNIRS = pd.read_csv(self.data_dir_fNIRS + self.file3_fNIRS,sep='\t')
        ## Store data into dictionary
        n302 = {'p1':p1, 'p2':p2, 'nw1':nw1, 'nw2':nw2}
        self.cohort = {'302':n302}
        self.cohort['302']['p1'].fs = self.fs_ecg
        ## Array with interval information
        intervals = self.setInterval(30, 5, self.fs_ecg, self.cohort['302']['p1'].shape[0])
        ## Extract different info such as lfhf and sdnn
        pack = []
        ecg_out = []
        RR_store = []
        for idx, val in enumerate(intervals[:-1]):
            current_segment = self.cohort['302']['p1'][val[0]:val[1]].ecg
            fs = self.cohort['302']['p1'].fs
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
        # Array with interval information
        intervals = [None]*(total_interval_over_strides)
        for i in range(total_interval_over_strides):
            intervals[i] = [i * stride_length, i * stride_length + interval_length]
        return intervals     
        
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


'''
Execute from here, for testing
'''
def main():
    processor = Processor()
    # lfhf_list = processor.getLfhf()
    # sdnn_list = processor.getSdnn()
    # sd_fq_array = processor.get_sdnn_lfhf_array()
    # theed_arr = processor.get3d()
    # tm = processor.getThreadMData()
    # print(lfhf_list)
    # print(sdnn_list)
    # print(tm)

if __name__=="__main__": 
    main()
    



