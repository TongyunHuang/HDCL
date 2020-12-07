# Entry point for accessing daya
from process_data import Processor

def get_single_trial_data(participant, trial):
    data_dict = {}
    Data = Processor(participant,trial)
    data_dict['data'] = Data
    data_dict['lfhf'] = Data.getLfhf()
    data_dict['sdnn'] = Data.getSdnn()
    data_dict['sd_lf'] = Data.get_sdnn_lfhf_array()
    data_dict['3d_dict'] = Data.get3d()
    data_dict['treadMill'] = Data.getTreadMData()
    data_dict['interval'] = Data.getInterval()
    print("Finish getting data for: " + trial)
    return data_dict

# Can be easily extend to receive several participant
def get_all_trial_seperate(participant):
    data_dict = {participant: { 'p1':{} , 'p2':{} , 'nw1':{} , 'nw2':{} } }
    for trial in data_dict[participant]:
        data_dict[participant][trial] = get_single_trial_data(participant, trial)
    return data_dict

# Can be easily extend to receive several participant
# def get_all_trial_cumulative(participant):
#     cumu_dict = {}
#     data_dict = get_all_trial_seperate(participant)

def main():
    data = get_all_trial_seperate('302')
    print(data)
    
    
    

if __name__=="__main__": 
    main()
