import os
import pandas as pd
import pickle
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression, LinearRegression
#from scipy import io
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns

from network import LSTM_classification, LSTM_regression
from utils import print_model_info, get_change_points

# drow Positive/Negative plot for each area
def convert_classification_results(x):
    if x.pred == 1:
        return "Positive"
    elif x.pred == 0:
        return "Negative"

# drow 10-level scale for each area (non-linear)
def convert_regression_results(x):
    if x.pred < 100*100:
        return "pred < 100*100"
    elif 100*100 <= x.pred < 200*200:
        return "100*100 <= pred < 200*200"
    elif 200*200 <= x.pred < 300*300:
        return "200*200 <= pred < 300*300"
    elif 300*300 <= x.pred < 400*400:
        return "300*300 <= pred < 400*400"
    elif 400*400 <= x.pred < 500*500:
        return "400*400 <= pred < 500*500" 
    elif 500*500 <= x.pred < 600*600:
        return "500*500 <= pred < 600*600"
    elif 600*600 <= x.pred < 700*700:
        return "600*600 <= pred < 700*700"
    elif 700*700 <= x.pred < 800*800:
        return "700*700 <= pred < 800*800"
    elif 800*800 <= x.pred < 900*900:
        return "800*800 <= pred < 900*900"
    elif 900*900 <= x.pred < 1000*1000:
        return "900*900 <= pred < 1000*1000"

def future_predict(cfg):

    #################
    # unpack config #
    #################
    DATA_DIR = cfg.DATA_DIR
    MODEL_DIR = cfg.MODEL_DIR
    RESULT_DIR = cfg.RESULT_DIR
    PREDICT_DIR = cfg.PREDICT_DIR
    area_list = cfg.area_list
    feature_list = cfg.feature_list
    model_list = cfg.model_list
    train_end_period_list = cfg.train_end_period_list
    PROBLEM = cfg.PROBLEM
    DURATION = cfg.DURATION
    PROB_THRESHOLD = cfg.PROB_THRESHOLD
    MASKING = cfg.MASKING
    MASKING_THRESHOLD = cfg.MASKING_THRESHOLD
    INPUT_FEATURE_PERIOD_FOR_PREDICTION = cfg.INPUT_FEATURE_PERIOD_FOR_PREDICTION

    mesh_ids = cfg.mesh_ids

    ########################
    # load hyperparameters #
    ########################
    hyperparams = pd.read_csv(os.path.join(MODEL_DIR, 'best_hyperparams_'+PROBLEM+'.csv'), header=None, index_col=0)
    
    UNIT_NUMS = [int(hyperparams.loc['net__module__unit_num1']), int(hyperparams.loc['net__module__unit_num2']), int(hyperparams.loc['net__module__unit_num3']), int(hyperparams.loc['net__module__unit_num4'])]
    BATCH_SIZE = int(hyperparams.loc['net__batch_size'])
    LEARNING_RATE = float(hyperparams.loc['net__optimizer__lr'])
    WEIGHT_DECAY = float(hyperparams.loc['net__optimizer__weight_decay'])
    DROPOUT_RATE = cfg.DROPOUT_RATE

    ####################################
    # predict future deforestation map #
    ####################################
    # create input data for prediction
    print('==========================')
    print('pred period: ', INPUT_FEATURE_PERIOD_FOR_PREDICTION)

    preds = np.zeros([mesh_ids.shape[0], len(model_list)]) # (N, N_model)

    # check change points of mesh ids over multiple areas
    change_points = get_change_points(mesh_ids)

    for model_id in range(len(model_list)):
        MODEL_NAME = model_list[model_id]
        #print('model: ', MODEL_NAME)
            
        for area_id in range(len(area_list)):
            target_area = area_list[area_id]
        
            for feature_id in range(len(feature_list)):
                temp_features = pd.read_csv(os.path.join(DATA_DIR, target_area, 'feature', feature_list[feature_id]+'.csv'), header=0)
                input_feature_period_id = temp_features.columns.get_loc(INPUT_FEATURE_PERIOD_FOR_PREDICTION)

                temp_features = np.array(temp_features)
                temp_features = temp_features[:,input_feature_period_id-DURATION+1:input_feature_period_id+1]
                temp_features = np.expand_dims(temp_features, -1)

                if feature_id==0:
                    features = temp_features
                else:
                    features = np.concatenate([features, temp_features], axis=2) # (N, L, D)
        
            # change variable name and type
            X = features.astype(np.float32)

            # standardization by prepared scalar (mu and sigma) in training phase
            scalar = pickle.load(open(os.path.join(MODEL_DIR, "scalar_"+train_end_period_list[-1]+"_"+PROBLEM+".sav"), "rb"))
            mu, sigma = scalar["mu"], scalar["sigma"]
            X = (X - mu) / sigma
            #print('mean: ', np.mean(X, axis=0))
            #print('std: ', np.std(X, axis=0))

            # make 2D input data for logistic/linear regression
            X_2d = X.reshape(X.shape[0], X.shape[1]*X.shape[2])

            # construct network and logistic/linear regressor
            INPUT_DIM = X.shape[2]
            if PROBLEM=='classification':
                OUTPUT_DIM = 2
                net = LSTM_classification(INPUT_DIM, UNIT_NUMS, OUTPUT_DIM, DROPOUT_RATE)
                lr_model = LogisticRegression(penalty='none')
            elif PROBLEM=='regression':
                OUTPUT_DIM = 1
                net = LSTM_regression(INPUT_DIM, UNIT_NUMS, OUTPUT_DIM, DROPOUT_RATE)
                lr_model = LinearRegression()
            #print_model_info(net)

            # load weights of the latest models
            net.load_state_dict(torch.load(os.path.join(MODEL_DIR, "LSTM_"+train_end_period_list[-1]+"_"+PROBLEM+".pytorch")))
            lr_model = pickle.load(open(os.path.join(MODEL_DIR, "LR_"+train_end_period_list[-1]+"_"+PROBLEM+".sav"), "rb"))

            # load the sum of deforestation area
            npz = np.load(os.path.join(RESULT_DIR, 'deforestation_sum_'+train_end_period_list[-1]+".npz"))
            deforestation_sum = npz['arr_0']

            # set eval mode
            net.eval()
            device_cpu = torch.device("cpu")
            net.to(device_cpu)
                
            # deforestation prediction
            if MODEL_NAME=='LSTM':
                # change type to torch.tensor type
                X = torch.tensor(X).float()
                area_preds = net(X)
                if PROBLEM=='classification':
                    area_preds = area_preds.detach().cpu().numpy()
                    pos_prob = area_preds[:,-1] # store probability of positive class
                    area_preds = np.argmax(area_preds, axis=1)
                    area_preds = np.where(pos_prob<PROB_THRESHOLD, 0, area_preds) # mask predictions based on threshold to probability
                elif PROBLEM=='regression':
                    area_preds = area_preds.detach().cpu().numpy()
            elif MODEL_NAME=='LR':
                area_preds = lr_model.predict(X_2d)
            
            # apply masking process based on the sum of deforestation area
            if MASKING:
                mask_id = np.where(MASKING_THRESHOLD<=deforestation_sum[change_points[area_id]:change_points[area_id+1]])
                area_preds[mask_id[0],] = 0
    
            # generate prediction map
            if PROBLEM=='classification':
                gdf_mesh_info = gpd.read_file(os.path.join(DATA_DIR, area_list[area_id], "label", "1km_mesh.geojson"))
            elif PROBLEM=='regression':
                gdf_mesh_info = gpd.read_file(os.path.join(DATA_DIR, area_list[area_id], "label", "1km_mesh_deforestation_square_meter.geojson"))
            gdf_mesh_info = gdf_mesh_info.to_crs(4326)

            df_area_preds = pd.DataFrame(area_preds)
            df_area_preds = df_area_preds.set_axis(['pred'], axis=1)

            gdf_area = gpd.GeoDataFrame(pd.concat([df_area_preds,gdf_mesh_info], axis=1),crs=gdf_mesh_info.crs)
            gdf_area["result"] = np.nan

            if PROBLEM=='classification':
                gdf_area["result"] = gdf_area.apply(convert_classification_results, axis=1)
                palette = {'Positive':'red', 'Negative': 'grey'}
                gdf_area["Colors"] = gdf_area["result"].map(palette)

                custom_points = [Line2D([0], [0], marker="s", linestyle="none", markersize=10, color=color) for color in palette.values()]
                ax = gdf_area.plot(color=gdf_area["Colors"])
                ax.legend(custom_points, palette.keys(), bbox_to_anchor=(1, 1), loc='upper left')
            if PROBLEM=='regression':
                gdf_area["result"] = gdf_area.apply(convert_regression_results, axis=1)
                palette = {'pred < 100*100': 'snow', '100*100 <= pred < 200*200': 'silver', 
                           '200*200 <= pred < 300*300': 'gray', '300*300 <= pred < 400*400': 'red',
                           '400*400 <= pred < 500*500': 'tomato', '500*500 <= pred < 600*600': 'lightsalmon', 
                           '600*600 <= pred < 700*700': 'green', '700*700 <= pred < 800*800': 'lime',
                           '800*800 <= pred < 900*900': 'blue', '900*900 <= pred < 1000*1000': 'cyan'}      
                gdf_area["Colors"] = gdf_area["result"].map(palette)
 
                custom_points = [Line2D([0], [0], marker="s", linestyle="none", markersize=10, color=color) for color in palette.values()]
                ax = gdf_area.plot(color=gdf_area["Colors"])
                ax.legend(custom_points, palette.keys(), bbox_to_anchor=(1, 1), loc='upper left')
                
            plt.subplots_adjust(right=0.8)
            plt.title(area_list[area_id]+"_"+ INPUT_FEATURE_PERIOD_FOR_PREDICTION)
            plt.tight_layout()
            plt.savefig(os.path.join(PREDICT_DIR, area_list[area_id]+"_"+INPUT_FEATURE_PERIOD_FOR_PREDICTION+"_"+MODEL_NAME+"_"+PROBLEM+".png"), dpi=400, facecolor="white", bbox_inches='tight') 
            plt.close()

            # save gdf_area as geojson file
            gdf_area.to_file(driver='GeoJSON', filename=os.path.join(PREDICT_DIR, area_list[area_id]+"_"+INPUT_FEATURE_PERIOD_FOR_PREDICTION+"_"+MODEL_NAME+"_"+PROBLEM+".geojson"))

            # concatenate preds
            if area_id==0:
                temp_preds = area_preds
            else:
                temp_preds = np.append(temp_preds, area_preds, axis=0)
        
        if PROBLEM=='regression':
            temp_preds = np.squeeze(temp_preds)

        preds[:,model_id] = temp_preds
        
    # concatenate mesh ids to predicted/true results
    mesh_ids_dammy = mesh_ids[:,np.newaxis] 
    preds = np.concatenate([mesh_ids_dammy, preds], axis=1)

    # save prediction as npz file
    np.savez(os.path.join(PREDICT_DIR, 'preds_'+PROBLEM), preds)
    #io.savemat(os.path.join(PREDICT_DIR, 'preds_'+PROBLEM+'.mat'), {"preds": preds})