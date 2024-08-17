import os
import random
import numpy as np
import torch
from skorch import NeuralNetClassifier, NeuralNetRegressor
from skorch.callbacks import Callback, EarlyStopping
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, KFold
import pandas as pd
import csv

from network import LSTM_skorch_classification, LSTM_skorch_regression
from utils import print_model_info, get_change_points

import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)

def grid_search(cfg):

    #################
    # unpack config #
    #################
    SEED = cfg.SEED
    CV_NUM = cfg.CV_NUM
    EPOCH_NUM = cfg.EPOCH_NUM
    PATIENCE = cfg.PATIENCE
    DROPOUT_RATE = cfg.DROPOUT_RATE
    WORKER_NUM = cfg.WORKER_NUM
    SCORE = cfg.SCORE

    unit_num_range = cfg.unit_num_range
    batch_size_range = cfg.batch_size_range
    lr_range = cfg.lr_range
    weight_decay_range = cfg.weight_decay_range

    MODEL_DIR = cfg.MODEL_DIR
    area_list = cfg.area_list
    test_period_list = cfg.test_period_list
    PROBLEM = cfg.PROBLEM

    period_info = cfg.period_info
    mesh_ids = cfg.mesh_ids
    labels = cfg.labels
    features = cfg.features

    ###################
    # fix random seed #
    ###################
    os.environ['PYTHONHASHSEED'] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.use_deterministic_algorithms = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    ####################################
    # create balanced training dataset #
    ####################################
    # find index of first testing period in entire dataset
    first_test_id = period_info.index(test_period_list[0])
    
    # extract training dataset
    train_labels = labels[:,:first_test_id]         # (N, P_train)
    train_features = features[:,:first_test_id,:,:] # (N, P_train, L, D)
    #print('train_features shape: ', train_features.shape)

    # buffer for balanced training dataset
    balanced_train_labels, balanced_train_features = [], []

    # check change points of mesh ids over multiple areas
    change_points = get_change_points(mesh_ids)
    #print("change_points: ", change_points)

    for area_id in range(len(area_list)):
        # extract training labels and features of the area
        #area_ids = mesh_ids[change_points[area_id]:change_points[area_id+1]]
        #print('area: ', area_list[area_id])
        #print('area_ids: ', area_ids)
        area_train_labels = train_labels[change_points[area_id]:change_points[area_id+1],:]
        area_train_features = train_features[change_points[area_id]:change_points[area_id+1],:]
        #print('area_train_labels: ', area_train_labels.shape)
        #print('area_train_features: ', area_train_features.shape)

        # reshape training labels and features
        area_train_labels = area_train_labels.reshape(area_train_labels.shape[0]*area_train_labels.shape[1])                                                                     # (N*P_train, )
        area_train_features = area_train_features.reshape(area_train_features.shape[0]*area_train_features.shape[1], area_train_features.shape[2], area_train_features.shape[3]) # (N*P_train, L, D)

        # find positive data where deforestation has occurred
        positive_ids = np.where(area_train_labels>0)
        positive_labels = area_train_labels[positive_ids[0]]
        positive_features = area_train_features[positive_ids[0],:,:]

        # extract as many negatives as positives
        negative_ids = np.where(area_train_labels==0)
        negative_ids = rng.choice(negative_ids[0], len(positive_ids[0]), replace=False, axis=0)
        negative_labels = area_train_labels[negative_ids]
        negative_features = area_train_features[negative_ids,:,:]

        temp_labels = np.concatenate([positive_labels, negative_labels], axis=0)
        temp_features = np.concatenate([positive_features, negative_features], axis=0)

        # concatenate labels and feats
        if area_id==0:
            balanced_train_labels = temp_labels
            balanced_train_features = temp_features
        else:
            balanced_train_labels = np.append(balanced_train_labels, temp_labels, axis=0)
            balanced_train_features = np.append(balanced_train_features, temp_features, axis=0)

    print('balanced_train_labels: ', balanced_train_labels.shape)
    print('balanced_train_features: ', balanced_train_features.shape)

    # change variable name and type
    X = balanced_train_features.astype(np.float32)
    if PROBLEM=='classification':
        y = balanced_train_labels.astype(np.int64)
    elif PROBLEM=='regression':
        y = balanced_train_labels.astype(np.float32)
        y = y.reshape(-1,1) # this operation is needed for regression

    #print('size y: ', y.shape)

    # standardization
    m = np.mean(X, axis=0)
    s = np.std(X, axis=0)
    s += 0.000001 # avoid zero devide
    X = (X - m) / s
    #print('mean: ', np.mean(X, axis=0))
    #print('std: ', np.std(X, axis=0))

    #####################
    # construct network #
    #####################
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #print("gpu or cpu: ", device)

    input_dim = X.shape[2]
    temp_unit_num1, temp_unit_num2, temp_unit_num3, temp_unit_num4 = 50, 50, 50, 50 # temporary variables
    if PROBLEM=='classification':
        output_dim = 2
        MyNet = LSTM_skorch_classification(input_dim, temp_unit_num1, temp_unit_num2, temp_unit_num3, temp_unit_num4, output_dim, DROPOUT_RATE)
        net = NeuralNetClassifier(
            MyNet,
            criterion=torch.nn.CrossEntropyLoss,
            optimizer=torch.optim.Adam,
            max_epochs=EPOCH_NUM,
            device=device,
            callbacks=[EarlyStopping(patience=PATIENCE)],
            verbose=False
        ) 
    elif PROBLEM=='regression':
        output_dim = 1
        MyNet = LSTM_skorch_regression(input_dim, temp_unit_num1, temp_unit_num2, temp_unit_num3, temp_unit_num4, output_dim, DROPOUT_RATE)
        net = NeuralNetRegressor(
            MyNet,
            criterion=torch.nn.MSELoss, # bad selection for zero-inflated model
            optimizer=torch.optim.Adam,
            max_epochs=EPOCH_NUM,
            device=device,
            callbacks=[EarlyStopping(patience=PATIENCE)],
            verbose=False
        ) 

    #print_model_info(MyNet)
    pipe = Pipeline([
        ('net', net),
    ])

    ###############
    # grid search #
    ###############
    param_grid = [{'net__module__unit_num1': [unit_num1], 'net__module__unit_num2': [unit_num2], 'net__module__unit_num3': [unit_num3], 'net__module__unit_num4': [unit_num4], 'net__batch_size': [batch_size], 'net__optimizer__lr': [lr],  'net__optimizer__weight_decay': [weight_decay], 
                   'net__module__input_dim': [input_dim], 'net__module__output_dim': [output_dim], 'net__module__dropout_rate': [DROPOUT_RATE]}
                   for unit_num1 in unit_num_range for unit_num2 in unit_num_range if (unit_num2 <= unit_num1) for unit_num3 in unit_num_range if (unit_num3 <= unit_num2) for unit_num4 in unit_num_range if (unit_num4 <= unit_num3) for batch_size in batch_size_range for lr in lr_range for weight_decay in weight_decay_range]
    
    grid = GridSearchCV(pipe, param_grid, scoring=SCORE, verbose=1, cv=KFold(n_splits=CV_NUM, shuffle=False), n_jobs=WORKER_NUM)
    grid_result = grid.fit(X, y)

    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    #means = grid_result.cv_results_['mean_test_score']
    #stds = grid_result.cv_results_['std_test_score']
    #params = grid_result.cv_results_['params']
    #for mean, stdev, param in zip(means, stds, params):
    #    print("%f (%f) with: %r" % (mean, stdev, param))

    # find best hyperparameters
    best_score = [grid_result.best_score_]
    best_hyperparams = grid_result.best_params_

    print('best score: ', best_score)
    print('best hyperparams: ', best_hyperparams)

    # save results of hyperparameter search
    grid_result = pd.DataFrame.from_dict(grid_result.cv_results_)
    grid_result.to_csv(os.path.join(MODEL_DIR, 'grid_result_'+PROBLEM+'.csv'))

    # save hyperparameters
    with open(os.path.join(MODEL_DIR, 'best_hyperparams_'+PROBLEM+'.csv'), 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        for k, v in best_hyperparams.items():
            writer.writerow([k, v])