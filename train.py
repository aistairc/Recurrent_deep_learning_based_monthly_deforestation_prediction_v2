import os
import random
import numpy as np
import torch
import pandas as pd
import math
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
import torch
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle

from network import LSTM_classification, LSTM_regression
from utils import print_model_info, DataSet, get_change_points

import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)

class EarlyStopping:
    def __init__(self, patience=5, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
    
    def __call__(self, val_loss):
        score = val_loss

        if self.best_score is None:
            self.best_score = score
        elif score > self.best_score:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


def fit(cfg):

    #################
    # unpack config #
    #################
    SEED = cfg.SEED
    EPOCH_NUM = cfg.EPOCH_NUM
    PATIENCE = cfg.PATIENCE
    DROPOUT_RATE = cfg.DROPOUT_RATE
    
    MODEL_DIR = cfg.MODEL_DIR
    area_list = cfg.area_list
    train_end_period_list = cfg.train_end_period_list
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
    torch.backends.cudnn.benchmark = False

    ########################
    # load hyperparameters #
    ########################
    hyperparams = pd.read_csv(os.path.join(MODEL_DIR, 'best_hyperparams_'+PROBLEM+'.csv'), header=None, index_col=0)
    
    UNIT_NUMS = [int(hyperparams.loc['net__module__unit_num1']), int(hyperparams.loc['net__module__unit_num2']), int(hyperparams.loc['net__module__unit_num3']), int(hyperparams.loc['net__module__unit_num4'])]
    BATCH_SIZE = int(hyperparams.loc['net__batch_size'])
    LEARNING_RATE = float(hyperparams.loc['net__optimizer__lr'])
    WEIGHT_DECAY = float(hyperparams.loc['net__optimizer__weight_decay'])

    ############
    # training #
    ############
    for train_end_period in train_end_period_list:
        print('==========================')
        print('train end period: ', train_end_period)
    
        ####################################
        # create balanced training dataset #
        ####################################
        # find index of train end period in entire dataset
        train_end_id = period_info.index(train_end_period)
        #print('train end period: ', period_info[:train_end_id+1])

        # extract training dataset
        train_labels = labels[:,:train_end_id+1]         # (N, P_train)
        train_features = features[:,:train_end_id+1,:,:] # (N, P_train, L, D)
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
        #print('mean y: ', np.mean(y))

        # standardization
        mu = np.mean(X, axis=0)
        sigma = np.std(X, axis=0)
        sigma += 0.000001 # avoid zero devide
        X = (X - mu) / sigma
        #print('mean: ', np.mean(X, axis=0))
        #print('std: ', np.std(X, axis=0))

        # devide train and val dataset into 8:2
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=math.floor(len(X)*0.2), random_state=SEED)
        #print('train size:', len(X_train))
        #print('val size:', len(X_val))
        
        # make 2D input data for logistic/linear regression
        X_train_2d = X_train.reshape(X_train.shape[0], X_train.shape[1]*X_train.shape[2])

        ###################################################
        # construct network and logistic/linear regressor #
        ###################################################
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #print("gpu or cpu: ", device)

        INPUT_DIM = X.shape[2]
        if PROBLEM=='classification':
            OUTPUT_DIM = 2
            net = LSTM_classification(INPUT_DIM, UNIT_NUMS, OUTPUT_DIM, DROPOUT_RATE)
            dtype = torch.int64
            criterion = torch.nn.CrossEntropyLoss()
            lr_model = LogisticRegression(penalty='none')
        elif PROBLEM=='regression':
            OUTPUT_DIM = 1
            net = LSTM_regression(INPUT_DIM, UNIT_NUMS, OUTPUT_DIM, DROPOUT_RATE)
            dtype = torch.float32
            criterion = torch.nn.MSELoss() # bad selection for zero-inflated model
            lr_model = LinearRegression()
        #print_model_info(net)

        # set DataLoaders for mini-batch learning
        train_dataset = DataSet(X_train, y_train, transform=transforms.ToTensor())
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
            )
        val_dataset = DataSet(X_val, y_val, transform=transforms.ToTensor())
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
            )
        
        ##################
        # start training #
        ##################
        net.to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        early_stopping = EarlyStopping(patience=PATIENCE, verbose=True)
        
        minibatch_train_loss_list, minibatch_val_loss_list = [], []
        epoch_train_loss_list, epoch_val_loss_list = [], []

        for epoch in range(EPOCH_NUM):
            minibatch_train_loss, minibatch_val_loss = 0, 0

            net.train()
            with tqdm(enumerate(train_dataloader, 0), total=len(train_dataloader)) as pbar:
                pbar.set_description("[Epoch (train) %d/%d]" % (epoch + 1, EPOCH_NUM))
                for count, item in pbar:
                    mini_x, mini_y = item
                    mini_x, mini_y = mini_x.to(device), mini_y.to(device)
                    mini_y = mini_y.flatten()
                    mini_y = torch.tensor(mini_y, dtype=dtype)

                    optimizer.zero_grad()
                    outputs = net(mini_x)
                    loss = criterion(outputs, mini_y)
                    loss.backward()
                    optimizer.step()

                    minibatch_train_loss += loss.detach().cpu().numpy()
                    minibatch_train_loss_list.append(loss.detach().cpu().numpy())
                epoch_train_loss_list.append(minibatch_train_loss / count)
            
            net.eval()
            with tqdm(enumerate(val_dataloader, 0), total=len(val_dataloader)) as pbar:
                pbar.set_description("[Epoch (val) %d/%d]" % (epoch + 1, EPOCH_NUM))
                for count, item in pbar:
                    mini_x, mini_y = item
                    mini_x, mini_y = mini_x.to(device), mini_y.to(device)
                    mini_y = mini_y.flatten()
                    mini_y = torch.tensor(mini_y, dtype=dtype)

                    outputs = net(mini_x)
                    loss = criterion(outputs, mini_y)

                    minibatch_val_loss += loss.detach().cpu().numpy()
                    minibatch_val_loss_list.append(loss.detach().cpu().numpy())
                val_loss = minibatch_val_loss / count
                epoch_val_loss_list.append(minibatch_val_loss / count)
            
            early_stopping(val_loss)
            if early_stopping.early_stop:
                break
        
        # draw the graph for loss by batch and epoch
        plt.figure(figsize=[12,4])
        plt.subplot(1,2,1)
        plt.plot(minibatch_train_loss_list)
        plt.subplot(1,2,2)
        plt.plot(epoch_train_loss_list, "o--", label="train")
        plt.plot(epoch_val_loss_list, "o--", label="val")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(MODEL_DIR, train_end_period+"_loss_"+PROBLEM+".jpg"), dpi=300)
        plt.clf()
        plt.close()

        # train logistic/linear regression model
        lr_model.fit(X_train_2d, y_train)

        # save mean and standard deviation
        scaler = {"sigma": sigma, "mu": mu}
        pickle.dump(scaler, open(os.path.join(MODEL_DIR, "scalar_"+train_end_period+"_"+PROBLEM+".sav"), "wb"))

        # save models
        torch.save(net.state_dict(), os.path.join(MODEL_DIR, "LSTM_"+train_end_period+"_"+PROBLEM+".pytorch"))
        pickle.dump(lr_model, open(os.path.join(MODEL_DIR, "LR_"+train_end_period+"_"+PROBLEM+".sav"), "wb"))