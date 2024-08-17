import os
import pandas as pd
import numpy as np
#from scipy import io

def make_dataset(cfg):
    # get period information, mesh ids, response variable (label), and explanatory variables (features)
    """
    N: total mesh num
    P: entire period length
    L: sequence length for prediction (=DURATION)
    D: feature dimension

    period_info: list        (P-L, )
    mesh_ids   : numpy array (N, )
    labels     : numpy array (N, P-L)
    features   : numpy array (N, P-L, L, D)
    """

    #################
    # unpack config #
    #################
    DATA_DIR = cfg.DATA_DIR
    area_list = cfg.area_list
    feature_list = cfg.feature_list
    entire_period_list = cfg.entire_period_list
    OBJECTIVE_VARIABLE = cfg.OBJECTIVE_VARIABLE
    PROBLEM = cfg.PROBLEM
    DURATION = cfg.DURATION

    for area_id in range(len(area_list)):
        print('area: ', area_list[area_id])

        #########################
        # get response variable #
        #########################
        target_area = area_list[area_id]
        temp_labels = pd.read_csv(os.path.join(DATA_DIR, target_area, 'label', OBJECTIVE_VARIABLE+'.csv'), header=0)
        label_period_start_ind = temp_labels.columns.get_loc(entire_period_list[0])
        label_period_end_ind = temp_labels.columns.get_loc(entire_period_list[-1])
        
        area_labels = np.array(temp_labels)
        area_mesh_ids = area_labels[:,0].astype(int)
        area_labels = area_labels[:,label_period_start_ind:label_period_end_ind+1] # extract target period
        area_labels = np.delete(area_labels, np.s_[:DURATION], 1)                  # remove onset period

        #############################
        # get explanatory variables #
        #############################
        for feature_id in range(len(feature_list)):
            temp_features = pd.read_csv(os.path.join(DATA_DIR, target_area, 'feature', feature_list[feature_id]+'.csv'), header=0)
            feature_period_start_ind = temp_features.columns.get_loc(entire_period_list[0])
            feature_period_end_ind = temp_features.columns.get_loc(entire_period_list[-1])
            temp_features = np.array(temp_features)

            for seq_id in range(feature_period_start_ind, feature_period_end_ind-(DURATION-1)):
                temp_seq = temp_features[:,seq_id:seq_id+DURATION]
                temp_seq = np.expand_dims(temp_seq, 1)
                temp_seq = np.expand_dims(temp_seq, -1)

                if seq_id==feature_period_start_ind:
                    temp_seq_features = temp_seq
                else:
                    temp_seq_features = np.concatenate([temp_seq_features, temp_seq], axis=1)
                
            if feature_id==0:
                area_features = temp_seq_features
            else:
                area_features = np.concatenate([area_features, temp_seq_features], axis=3)
        
        ###################################################################
        # concatinate response/explanatory variables from different areas #
        ###################################################################
        if area_id==0:
            period_info = list(temp_labels.columns.values[label_period_start_ind:label_period_end_ind+1])
            del period_info[:DURATION]
            mesh_ids = area_mesh_ids
            labels = area_labels
            features = area_features
        else:
            mesh_ids = np.concatenate([mesh_ids, area_mesh_ids], axis=0)
            labels = np.concatenate([labels, area_labels], axis=0)
            features = np.concatenate([features, area_features], axis=0)

    #############
    # save data #
    #############
    np.savez(os.path.join(DATA_DIR,'dataset_'+PROBLEM), period_info, mesh_ids, labels, features)
    #io.savemat(os.path.join(DATA_DIR,'dataset_'+PROBLEM+'.mat'), {"period_info": period_info, "mesh_ids": mesh_ids, "labels": labels, "features": features})