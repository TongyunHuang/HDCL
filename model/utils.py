import pandas as pd
import numpy as np

def load_data(file_name,data_dir='csv', skip=0,gdrive=True):
    file = data_dir*gdrive +  file_name
    df = pd.read_csv(file,skiprows = skip)
    return df

def delsys_cleanup(df, column='L PECTORALIS MAJOR: EKG 16'):
    'this function cleans up data that is coming from the delsys system'
    if column == 'all':
        df = df.fillna(0.0)
        start = np.where(np.diff(df)!=0)[0][0]+1
        end = np.where(np.diff(df[-1:0:-1])!=0)[0][0]+1
        df = df[start:-end].reset_index(drop=True)
        out = pd.concat([df.round(5), df],axis=1)
        
    else:
        df = df.fillna(0.0)
        start = np.where(np.diff(df[column])!=0)[0][0]+1
        end = np.where(np.diff(df[column][-1:0:-1])!=0)[0][0]+1
        ecg = df[column][start:-end].reset_index(drop=True)
        ecg_t = df.iloc[:, df.columns.get_loc(column)-1][start:-end].reset_index(drop=True)
        out = pd.concat([ecg_t.round(5), ecg],axis=1)
        out.columns = ['time','ecg']
        #out['time'] = pd.to_datetime(out['time'], unit = 's')
    return out
    
def resample(all_data, oversampling_period, undersampling_period, time_col = 'time'):
    """
    all_data: df with trajectories
    oversampling_period : increase resolution in time by interpolating within higher res period
    undersampling_period : picking points from the oversampled tajectory 
    tim_col 
    taken from eyas alfaris, theaeros github
    """
    traj_df =  all_data.copy()
    traj_df.set_index('time',inplace =True)
    
    tmp = traj_df.resample(oversampling_period).mean().interpolate() # oversample
    resampled = tmp.resample(undersampling_period).ffill()           # then undersample
    resampled.reset_index(inplace =True)
    #resampled = resampled.drop(['time'],axis= 1)
      
    return resampled

def cutting(input,dt):
    """
    this code cute the data and shift the index and the timesteps based on the time difference between two local time.
    dt caculated based on the time sync .txt files,
    input is the data you would like to cut
    """
    pos = next(x for x, val in enumerate(input['time']) if val >= dt) 
    out = input[pos::]
    out.reset_index(inplace =True)
    out = out.drop(columns=['index'])
    out['time'] = out['time']-dt
    return out
    
    
def cut_plz(something):
    fNIRS = something['fNIRS']
    emg_and_ecg = something['emg_and_ecg']
    acc = something['acc']
    trial = something['trial']
    
    if t_emg > t_tm and t_emg > t_fnirs: #, cut tm & fnirs

        #treadmill data
        dt = t_emg - t_tm 
        treadmill_data = cutting(treadmill_data, dt)

        #fNIRS
        dt = t_emg - t_fnirs
        fNIRS[trial] = cutting(fNIRS[trial], dt)
        

    elif t_fnirs > t_tm and t_fnirs > t_emg:

        #treadmill data
        dt = t_fnirs - t_tm 
        treadmill_data = cutting(treadmill_data, dt)

        #emg, acc & ecg  
        dt = t_fnirs - t_emg
        acc[trial] = cutting(acc[trial], dt)
        emg_and_ecg[trial] = cutting(emg_and_ecg[trial], dt)

    elif t_tm > t_emg and t_tm > t_fnirs:
        #emg, acc & ecg  
        dt = t_tm - t_emg 
        acc[trial] = cutting(acc[trial], dt)
        emg_and_ecg[trial] = cutting(emg_and_ecg[trial], dt)

        #fNIRS
        dt = t_tm - t_fnirs
        fNIRS[trial] = cutting(fNIRS[trial], dt)

    out['fNIRS'] = fNIRS[trial]
    out['emg_and_ecg'] = emg_and_ecg[trial]
    out['acc'] = acc[trial]
    out['trial'] = something['trial']    
    out['subject_id'] = something['subject_id']
    return out
