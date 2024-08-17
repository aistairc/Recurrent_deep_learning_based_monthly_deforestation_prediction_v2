## Main script for deforestation prediction
# checked OS: WSL2 (Ubuntu 22.04)

# Suguru Kanoga, 16-Aug.-2024
#  Artificial Intelligence Research Center, National Institute of Advanced
#  Industrial Science and Technology (AIST)
#  E-mail: s.kanouga@aist.go.jp

# Updated points
# 1. improved flexibility of whole scripts
# 2. added config function for managing variables
# 3. added new feature from road information
# 4. added regression scenario (but not effective)
# 5. separated training and evaluation steps
# 6. added future predicition function
# 7. added two masking functions (based on output probability and sum of deforestated area in a mesh)
# 8. added EarlyStopping function
# 9. added save function for geojson file to visualize results in QGIS software

###################
# import packages #
###################
# public
import os
import numpy as np

# private
from config import *
from preprocess import make_dataset
from hyperparameter_search import grid_search
import train
from predict import future_predict
from evaluate import offline_analysis, summary_table, visualize_offline_results, map_visualize_offline_analysis

##############
# set config #
##############
cfg = config()

##############
# preprocess #
##############
# make dataset
if not os.path.exists(os.path.join(cfg.DATA_DIR, "dataset_"+cfg.PROBLEM+".npz")) or cfg.REMAKING:
    make_dataset(cfg)

# load dataset
npz = np.load(os.path.join(cfg.DATA_DIR, "dataset_"+cfg.PROBLEM+".npz"))
period_info, mesh_ids, labels, features = npz['arr_0'], npz['arr_1'], npz['arr_2'], npz['arr_3']
period_info = list(period_info)

print('problem: ', cfg.PROBLEM)
print('areas: ', cfg.area_list)
print('preriod of dataset: ', period_info)
print('length of period: ', len(period_info))
print('total mesh ids: ', mesh_ids.shape)
print('shape of total labels: ', labels.shape)
print('shape of total features: ', features.shape)

# add the dataset to config
cfg.period_info = period_info
cfg.mesh_ids = mesh_ids
cfg.labels = labels
cfg.features = features

#########################
# hyperparameter search #
#########################
if not os.path.exists(os.path.join(cfg.MODEL_DIR, "best_hyperparams_"+cfg.PROBLEM+".csv")):
    print('-----------------')
    print('start hyperparameter search')
    print('-----------------')
    grid_search(cfg)
else:
    print('-----------------')
    print('already searched')
    print('-----------------')

###########################################
# training with optimized hyperparameters #
###########################################
if not os.path.exists(os.path.join(cfg.MODEL_DIR, "LSTM_"+cfg.train_end_period_list[-1]+"_"+cfg.PROBLEM+".pytorch")):
    print('---------------')
    print('start training')
    print('---------------')
    train.fit(cfg)
elif cfg.RETRAINING==True:
    print('-----------------')
    print('start retraining')
    print('-----------------')
    train.fit(cfg)
else:
    print('-----------------')
    print('already trained')
    print('-----------------')

######################
# offline evaluation #
######################
if not os.path.exists(os.path.join(cfg.RESULT_DIR, "results_"+cfg.PROBLEM+".npz")):
    print('-----------------')
    print('start evaluation')
    print('-----------------')
    offline_analysis(cfg)
    summary_table(cfg)
    visualize_offline_results(cfg)
    map_visualize_offline_analysis(cfg)
elif cfg.REEVALUATION==True:
    print('-------------------')
    print('start reevaluation')
    print('-------------------')
    offline_analysis(cfg)
    summary_table(cfg)
    visualize_offline_results(cfg)
    map_visualize_offline_analysis(cfg)
else:
    print('-----------------')
    print('already evaluated')
    print('-----------------')

#####################
# furure prediction # 
#####################
# predict deforestation event of next month (no true values) using the latest model
if cfg.FUTURE_PREDICT == True:
    # future prediction (no true values)
    print('-------------------------')
    print(' start future prediction ')
    print('-------------------------')
    future_predict(cfg)