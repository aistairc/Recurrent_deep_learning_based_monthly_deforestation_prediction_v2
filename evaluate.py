import os
import pandas as pd
import pickle
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.metrics import mean_squared_error, r2_score
#from scipy import io
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import datetime
import geopandas as gpd

from network import LSTM_classification, LSTM_regression
from utils import print_model_info, get_change_points

def offline_analysis(cfg):

    #################
    # unpack config #
    #################
    MODEL_DIR = cfg.MODEL_DIR
    RESULT_DIR = cfg.RESULT_DIR
    area_list = cfg.area_list
    feature_list = cfg.feature_list
    model_list = cfg.model_list
    train_end_period_list = cfg.train_end_period_list
    test_period_list = cfg.test_period_list
    PROBLEM = cfg.PROBLEM
    PROB_THRESHOLD = cfg.PROB_THRESHOLD
    MASKING = cfg.MASKING
    MASKING_THRESHOLD = cfg.MASKING_THRESHOLD

    period_info = cfg.period_info
    mesh_ids = cfg.mesh_ids
    labels = cfg.labels
    features = cfg.features
    
    ########################
    # load hyperparameters #
    ########################
    hyperparams = pd.read_csv(os.path.join(MODEL_DIR, 'best_hyperparams_'+PROBLEM+'.csv'), header=None, index_col=0)
    
    UNIT_NUMS = [int(hyperparams.loc['net__module__unit_num1']), int(hyperparams.loc['net__module__unit_num2']), int(hyperparams.loc['net__module__unit_num3']), int(hyperparams.loc['net__module__unit_num4'])]
    BATCH_SIZE = int(hyperparams.loc['net__batch_size'])
    LEARNING_RATE = float(hyperparams.loc['net__optimizer__lr'])
    WEIGHT_DECAY = float(hyperparams.loc['net__optimizer__weight_decay'])
    DROPOUT_RATE = cfg.DROPOUT_RATE

    ###########
    # testing #
    ###########
    # counter for testing period
    counter = 0

    # buffer for evaluation incecies
    preds = np.zeros([mesh_ids.shape[0], len(test_period_list), len(model_list)]) # (N, P_test, N_model)
    trues = np.zeros([mesh_ids.shape[0], len(test_period_list), len(model_list)])

    if PROBLEM=='classification':
        accs = np.zeros([len(area_list), len(test_period_list), len(model_list)]) # (N_area, P_test, N_model)
        recalls = np.zeros([len(area_list), len(test_period_list), len(model_list)])
        precs = np.zeros([len(area_list), len(test_period_list), len(model_list)])
        f1_scores = np.zeros([len(area_list), len(test_period_list), len(model_list)])
    elif PROBLEM=='regression':
        mses = np.zeros([len(area_list), len(test_period_list), len(model_list)])
        r2s = np.zeros([len(area_list), len(test_period_list), len(model_list)])

    #############
    # main loop #
    #############
    deforestation_sum = np.zeros(len(mesh_ids))

    for test_period in test_period_list:
        print('==========================')
        print('test period: ', test_period)

        ##########################
        # create testing dataset #
        ##########################
        test_id = period_info.index(test_period)
        target_id = period_info.index(test_period)-1
        #print('target period: ', period_info[target_id])

        test_labels = labels[:,test_id]         # (N, )
        test_features = features[:,test_id,:,:] # (N, L, D)

        # change variable name and type
        X = test_features.astype(np.float32)
        if PROBLEM=='classification':
            y = test_labels.astype(np.int64)
        elif PROBLEM=='regression':
            y = test_labels.astype(np.float32)
            y = y.reshape(-1,1) # this operation is needed for regression
        
        #print('size y: ', y.shape)

        # standardization by prepared scalar (mu and sigma) in training phase
        scalar = pickle.load(open(os.path.join(MODEL_DIR, "scalar_"+period_info[target_id]+"_"+PROBLEM+".sav"), "rb"))
        mu, sigma = scalar["mu"], scalar["sigma"]
        X = (X - mu) / sigma
        #print('mean: ', np.mean(X, axis=0))
        #print('std: ', np.std(X, axis=0))

        # make 2D input data for logistic/linear regression
        X_2d = X.reshape(X.shape[0], X.shape[1]*X.shape[2])
        
        ###################################################
        # construct network and logistic/linear regressor #
        ###################################################
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

        # load weights
        net.load_state_dict(torch.load(os.path.join(MODEL_DIR, "LSTM_"+period_info[target_id]+"_"+PROBLEM+".pytorch")))
        lr_model = pickle.load(open(os.path.join(MODEL_DIR, "LR_"+period_info[target_id]+"_"+PROBLEM+".sav"), "rb"))

        ############################################
        # evaluate model performance for each area #
        ############################################
        # set eval mode
        net.eval()
        device_cpu = torch.device("cpu")
        net.to(device_cpu)

        # check change points of mesh ids over multiple areas
        change_points = get_change_points(mesh_ids)
        #print("change_points: ", change_points)

        for model_id in range(len(model_list)):
            MODEL_NAME = model_list[model_id]
            #print('model: ', MODEL_NAME)
            
            for area_id in range(len(area_list)):
                #print('area: ', area_list[area_id])

                # extract X and y of the target area
                if MODEL_NAME=='LSTM':
                    area_X = X[change_points[area_id]:change_points[area_id+1],:,:]
                    # change type to torch.tensor type
                    area_X = torch.tensor(area_X).float()
                elif MODEL_NAME=='LR':
                    area_X_2d = X_2d[change_points[area_id]:change_points[area_id+1],:]
                
                if PROBLEM=='classification':
                    area_y = y[change_points[area_id]:change_points[area_id+1],]
                elif PROBLEM=='regression':
                    area_y = y[change_points[area_id]:change_points[area_id+1],:]

                # deforestation prediction
                if MODEL_NAME=='LSTM':
                    area_preds = net(area_X)
                    if PROBLEM=='classification':
                        area_preds = area_preds.detach().cpu().numpy()
                        pos_prob = area_preds[:,-1] # store probability of positive class
                        area_preds = np.argmax(area_preds, axis=1)
                        area_preds = np.where(pos_prob<PROB_THRESHOLD, 0, area_preds) # mask predictions based on threshold to probability
                    elif PROBLEM=='regression':
                        area_preds = area_preds.detach().cpu().numpy()
                elif MODEL_NAME=='LR':
                    area_preds = lr_model.predict(area_X_2d)

                # apply masking process based on the sum of deforestation area
                if MASKING:
                    mask_id = np.where(MASKING_THRESHOLD<=deforestation_sum[change_points[area_id]:change_points[area_id+1]])
                    area_preds[mask_id[0],] = 0

                #print('area_y shape: ', area_y.shape)
                #print('area_preds shape: ', area_preds.shape)

                # check performance in four indecies for classification
                if PROBLEM=='classification':
                    accs[area_id,counter,model_id] = accuracy_score(area_y, area_preds)
                    recalls[area_id,counter,model_id] = recall_score(area_y, area_preds, average="macro")
                    precs[area_id,counter,model_id] = precision_score(area_y, area_preds, average="macro")
                    f1_scores[area_id,counter,model_id] = f1_score(area_y, area_preds, average="macro")

                    #print(MODEL_NAME + ' accuracy: ', accs[area_id,counter,model_id])
                    #print(MODEL_NAME + ' recall: ', recalls[area_id,counter,model_id])
                    #print(MODEL_NAME + ' precision: ', precs[area_id,counter,model_id])
                    #print(MODEL_NAME + ' f1 score: ', f1_scores[area_id,counter,model_id])
                elif PROBLEM=='regression':
                    mses[area_id,counter,model_id] = mean_squared_error(area_y, area_preds)
                    r2s[area_id,counter,model_id] = r2_score(area_y, area_preds)

                    #print(MODEL_NAME + ' mse: ', mses[area_id,counter,model_id])
                    #print(MODEL_NAME + ' r2: ', r2s[area_id,counter,model_id])
                
                # concatenate preds and true values
                if area_id==0:
                    temp_preds = area_preds
                    temp_trues = area_y
                else:
                    temp_preds = np.append(temp_preds, area_preds, axis=0)
                    temp_trues = np.append(temp_trues, area_y, axis=0)
        
            if PROBLEM=='regression':
                temp_preds = np.squeeze(temp_preds)
                temp_trues = np.squeeze(temp_trues)

            preds[:,counter,model_id] = temp_preds
            trues[:,counter,model_id] = temp_trues
        counter = counter + 1

        # get/updata history of deforestated areas for each area
        if np.sum(deforestation_sum)==0:
            history = features[:,:test_id,-1,feature_list.index("deforestation_square_meter")] # (N, P_until_test)
            history = np.sum(history, axis=1) # (N,)
        else:
            history = test_features[:,-1,feature_list.index("deforestation_square_meter")] # (N, )
        
        deforestation_sum = deforestation_sum + history
        np.savez(os.path.join(RESULT_DIR, 'deforestation_sum_'+period_info[target_id]), deforestation_sum)
        #io.savemat(os.path.join(RESULT_DIR, 'deforestation_sum_'+period_info[target_id]+'.mat'), {"deforestation_sum": deforestation_sum})

    # concatenate mesh ids to predicted/true results
    mesh_ids_dammy = mesh_ids[:,np.newaxis] 
    mesh_ids_dammy = np.tile(mesh_ids_dammy,(1,len(model_list)))
    mesh_ids_dammy = mesh_ids_dammy[:,np.newaxis,:]
    preds = np.concatenate([mesh_ids_dammy, preds], axis=1)
    trues = np.concatenate([mesh_ids_dammy, trues], axis=1)

    # save results
    if PROBLEM=='classification':
        np.savez(os.path.join(RESULT_DIR, 'results_'+PROBLEM), preds, trues, accs, recalls, precs, f1_scores)
        #io.savemat(os.path.join(RESULT_DIR, 'results_'+PROBLEM+'.mat'), {"preds": preds, "trues": trues,
        #                                                                 "accs": accs, "recalls": recalls, "precs": precs, "f1_scores": f1_scores})
    elif PROBLEM=='regression':
        np.savez(os.path.join(RESULT_DIR, 'results_'+PROBLEM), preds, trues, mses, r2s)
        #io.savemat(os.path.join(RESULT_DIR, 'results_'+PROBLEM+'.mat'), {"preds": preds, "trues": trues,
        #                                                                 "mses": mses, "r2s": r2s})

def summary_table(cfg):

    #################
    # unpack config #
    #################
    RESULT_DIR = cfg.RESULT_DIR
    area_list = cfg.area_list
    model_list = cfg.model_list
    test_period_list = cfg.test_period_list
    eval_period_list = cfg.eval_period_list
    PROBLEM = cfg.PROBLEM

    ################
    # load results #
    ################
    npz = np.load(os.path.join(RESULT_DIR, 'results_'+PROBLEM+'.npz'))

    if PROBLEM=='classification':
        accs, recalls, precs, f1_scores = npz['arr_2'], npz['arr_3'], npz['arr_4'], npz['arr_5']
    elif PROBLEM=='regression':
        mses, r2s = npz['arr_2'], npz['arr_3']

    eval_ids = []
    for i in range(len(eval_period_list)):
        eval_ids.append(test_period_list.index(eval_period_list[i]))

    ########################
    # make a summary table #
    ########################
    print('==============')
    print('summary table')
    print('==============')
    for area_id in range(len(area_list)):
        print(area_list[area_id])
        
        if PROBLEM=='classification':
            print('           acc    recall   precision  f1 score')
            for model_id in range(len(model_list)):
                print(model_list[model_id]+'_mean:  {:.3f}   {:.3f}     {:.3f}       {:.3f}'.format(np.mean(accs[area_id,eval_ids,model_id]), np.mean(recalls[area_id,eval_ids,model_id]), np.mean(precs[area_id,eval_ids,model_id]), np.mean(f1_scores[area_id,eval_ids,model_id])))
                print(model_list[model_id]+'_max:   {:.3f}   {:.3f}     {:.3f}       {:.3f}'.format(np.max(accs[area_id,eval_ids,model_id]), np.max(recalls[area_id,eval_ids,model_id]), np.max(precs[area_id,eval_ids,model_id]), np.max(f1_scores[area_id,eval_ids,model_id])))
        elif PROBLEM=='regression':
            print('           mse    r2')
            for model_id in range(len(model_list)):
                print(model_list[model_id]+'_mean:  {:.3f}   {:.3f}'.format(np.mean(mses[area_id,eval_ids,model_id]), np.mean(r2s[area_id,eval_ids,model_id])))
                print(model_list[model_id]+'_max:   {:.3f}   {:.3f}'.format(np.max(mses[area_id,eval_ids,model_id]), np.max(r2s[area_id,eval_ids,model_id])))
        
        print('\n')

def visualize_offline_results(cfg):

    #################
    # unpack config #
    #################
    RESULT_DIR = cfg.RESULT_DIR
    area_list = cfg.area_list
    model_list = cfg.model_list
    test_period_list = cfg.test_period_list
    eval_period_list = cfg.eval_period_list
    PROBLEM = cfg.PROBLEM

    rain_start_year_list = cfg.rain_start_year_list
    rain_end_year_list = cfg.rain_end_year_list

    ###################
    # set figure info #
    ###################
    plt.rcParams['font.size'] = 14
    rain_start_datetime, rain_end_datetime = [], []
    for rain_id in range(len(rain_start_year_list)):
        rain_start_datetime.append(datetime.datetime(rain_start_year_list[rain_id], 10, 1))
        rain_end_datetime.append(datetime.datetime(rain_end_year_list[rain_id], 3, 1))

    ################
    # load results #
    ################
    npz = np.load(os.path.join(RESULT_DIR, 'results_'+PROBLEM+'.npz'))

    #################################################################
    # plot average results over all areas and results for each area #
    #################################################################
    if PROBLEM=='classification':
        fig = plt.figure(figsize=(15,10))
        accs, recalls, precs, f1_scores = npz['arr_2'], npz['arr_3'], npz['arr_4'], npz['arr_5']
        ax1, ax2, ax3, ax4 = fig.add_subplot(2,2,1), fig.add_subplot(2,2,2), fig.add_subplot(2,2,3), fig.add_subplot(2,2,4)

        # drow average results over all areas
        for model_id in range(len(model_list)):
            pd_accs = pd.DataFrame({"Val": np.mean(accs[:,:,model_id], axis=0)}, index=pd.Index(pd.date_range(test_period_list[0][0:4]+"/"+test_period_list[0][4:6]+"/01", periods=len(test_period_list), freq='MS'), name="Data"))
            pd_recalls = pd.DataFrame({"Val": np.mean(recalls[:,:,model_id], axis=0)}, index=pd.Index(pd.date_range(test_period_list[0][0:4]+"/"+test_period_list[0][4:6]+"/01", periods=len(test_period_list), freq='MS'), name="Data"))
            pd_precs = pd.DataFrame({"Val": np.mean(precs[:,:,model_id], axis=0)}, index=pd.Index(pd.date_range(test_period_list[0][0:4]+"/"+test_period_list[0][4:6]+"/01", periods=len(test_period_list), freq='MS'), name="Data"))
            pd_f1_scores = pd.DataFrame({"Val": np.mean(f1_scores[:,:,model_id], axis=0)}, index=pd.Index(pd.date_range(test_period_list[0][0:4]+"/"+test_period_list[0][4:6]+"/01", periods=len(test_period_list), freq='MS'), name="Data"))

            for fig_id in range(4):
                if fig_id == 0: # acc
                    ax1.set_ylim(0, 1)
                    ax1.plot(pd_accs, label=model_list[model_id], linewidth=2)
                    for rain_id in range(len(rain_start_year_list)):
                        ax1.axvspan(rain_start_datetime[rain_id], rain_end_datetime[rain_id], color="gray", alpha=0.3)
                    if not model_id==0:
                        ax1.legend(loc="upper right")
                        ax1.set_title("Monthly averaged acc")
                        ax1.grid()
                elif fig_id == 1: # recall
                    ax2.set_ylim(0, 1)
                    ax2.plot(pd_recalls, label=model_list[model_id], linewidth=2)
                    for rain_id in range(len(rain_start_year_list)):
                        ax2.axvspan(rain_start_datetime[rain_id], rain_end_datetime[rain_id], color="gray", alpha=0.3)
                    if not model_id==0:
                        ax2.legend(loc="upper left")
                        ax2.set_title("Monthly averaged recall")
                        ax2.grid()
                elif fig_id == 2: # precision
                    ax3.set_ylim(0, 1)
                    ax3.plot(pd_precs, label=model_list[model_id], linewidth=2)
                    for rain_id in range(len(rain_start_year_list)):
                        ax3.axvspan(rain_start_datetime[rain_id], rain_end_datetime[rain_id], color="gray", alpha=0.3)
                    if not model_id==0:
                        ax3.legend(loc="lower left")
                        ax3.set_title("Monthly averaged precision")
                        ax3.grid()
                elif fig_id == 3: # f1 score
                    ax4.set_ylim(0, 1)
                    ax4.plot(pd_f1_scores, label=model_list[model_id], linewidth=2)
                    for rain_id in range(len(rain_start_year_list)):
                        ax4.axvspan(rain_start_datetime[rain_id], rain_end_datetime[rain_id], color="gray", alpha=0.3)
                    if not model_id==0:
                        ax4.legend(loc="lower right")
                        ax4.set_title("Monthly averaged f1 score")
                        ax4.grid()
                plt.gcf().autofmt_xdate()
        fig.tight_layout()
        fig.savefig(os.path.join(RESULT_DIR,"average_result_classification.jpg"), dpi=300)
        fig.clf()

        # drow results for each area
        for area_id in range(len(area_list)):
            fig = plt.figure(figsize=(15,10))
            ax1, ax2, ax3, ax4 = fig.add_subplot(2,2,1), fig.add_subplot(2,2,2), fig.add_subplot(2,2,3), fig.add_subplot(2,2,4)

            for model_id in range(len(model_list)):
                pd_accs = pd.DataFrame({"Val": accs[area_id,:,model_id]}, index=pd.Index(pd.date_range(test_period_list[0][0:4]+"/"+test_period_list[0][4:6]+"/01", periods=len(test_period_list), freq='MS'), name="Data"))
                pd_recalls = pd.DataFrame({"Val": recalls[area_id,:,model_id]}, index=pd.Index(pd.date_range(test_period_list[0][0:4]+"/"+test_period_list[0][4:6]+"/01", periods=len(test_period_list), freq='MS'), name="Data"))
                pd_precs = pd.DataFrame({"Val": precs[area_id,:,model_id]}, index=pd.Index(pd.date_range(test_period_list[0][0:4]+"/"+test_period_list[0][4:6]+"/01", periods=len(test_period_list), freq='MS'), name="Data"))
                pd_f1_scores = pd.DataFrame({"Val": f1_scores[area_id,:,model_id]}, index=pd.Index(pd.date_range(test_period_list[0][0:4]+"/"+test_period_list[0][4:6]+"/01", periods=len(test_period_list), freq='MS'), name="Data"))

                for fig_id in range(4):
                    if fig_id == 0: # acc
                        ax1.set_ylim(0, 1)
                        ax1.plot(pd_accs, label=model_list[model_id], linewidth=2)
                        for rain_id in range(len(rain_start_year_list)):
                            ax1.axvspan(rain_start_datetime[rain_id], rain_end_datetime[rain_id], color="gray", alpha=0.3)
                        if not model_id==0:
                            ax1.legend(loc="upper right")
                            ax1.set_title("Monthly averaged acc")
                            ax1.grid()
                    elif fig_id == 1: # recall
                        ax2.set_ylim(0, 1)
                        ax2.plot(pd_recalls, label=model_list[model_id], linewidth=2)
                        for rain_id in range(len(rain_start_year_list)):
                            ax2.axvspan(rain_start_datetime[rain_id], rain_end_datetime[rain_id], color="gray", alpha=0.3)
                        if not model_id==0:
                            ax2.legend(loc="upper left")
                            ax2.set_title("Monthly averaged recall")
                            ax2.grid()
                    elif fig_id == 2: # precision
                        ax3.set_ylim(0, 1)
                        ax3.plot(pd_precs, label=model_list[model_id], linewidth=2)
                        for rain_id in range(len(rain_start_year_list)):
                            ax3.axvspan(rain_start_datetime[rain_id], rain_end_datetime[rain_id], color="gray", alpha=0.3)
                        if not model_id==0:
                            ax3.legend(loc="lower left")
                            ax3.set_title("Monthly averaged precision")
                            ax3.grid()
                    elif fig_id == 3: # f1 score
                        ax4.set_ylim(0, 1)
                        ax4.plot(pd_f1_scores, label=model_list[model_id], linewidth=2)
                        for rain_id in range(len(rain_start_year_list)):
                            ax4.axvspan(rain_start_datetime[rain_id], rain_end_datetime[rain_id], color="gray", alpha=0.3)
                        if not model_id==0:
                            ax4.legend(loc="lower right")
                            ax4.set_title("Monthly averaged f1 score")
                            ax4.grid()
                    plt.gcf().autofmt_xdate()
            fig.tight_layout()
            fig.savefig(os.path.join(RESULT_DIR, area_list[area_id]+"_result_classification.jpg"), dpi=300)
            fig.clf()

    elif PROBLEM=='regression':
        fig = plt.figure(figsize=(15,10))
        mses, r2s = npz['arr_2'], npz['arr_3']
        ax1, ax2 = fig.add_subplot(2,1,1), fig.add_subplot(2,1,2)

        # drow average results over all areas
        for model_id in range(len(model_list)):
            pd_mses = pd.DataFrame({"Val": np.mean(mses[:,:,model_id], axis=0)}, index=pd.Index(pd.date_range(test_period_list[0][0:4]+"/"+test_period_list[0][4:6]+"/01", periods=len(test_period_list), freq='MS'), name="Data"))
            pd_r2s = pd.DataFrame({"Val": np.mean(r2s[:,:,model_id], axis=0)}, index=pd.Index(pd.date_range(test_period_list[0][0:4]+"/"+test_period_list[0][4:6]+"/01", periods=len(test_period_list), freq='MS'), name="Data"))

            for fig_id in range(4):
                if fig_id == 0: # mse
                    ax1.plot(pd_mses, label=model_list[model_id], linewidth=2)
                    for rain_id in range(len(rain_start_year_list)):
                        ax1.axvspan(rain_start_datetime[rain_id], rain_end_datetime[rain_id], color="gray", alpha=0.3)
                    if not model_id==0:
                        ax1.legend(loc="upper right")
                        ax1.set_title("Monthly averaged mse")
                        ax1.grid()
                elif fig_id == 1: # r2
                    ax2.plot(pd_r2s, label=model_list[model_id], linewidth=2)
                    for rain_id in range(len(rain_start_year_list)):
                        ax2.axvspan(rain_start_datetime[rain_id], rain_end_datetime[rain_id], color="gray", alpha=0.3)
                    if not model_id==0:
                        ax2.legend(loc="upper right")
                        ax2.set_title("Monthly averaged r2")
                        ax2.grid()
                plt.gcf().autofmt_xdate()
        fig.tight_layout()
        fig.savefig(os.path.join(RESULT_DIR,"average_result_regression.jpg"), dpi=300)
        fig.clf()

        # drow results for each area
        for area_id in range(len(area_list)):
            fig = plt.figure(figsize=(15,10))
            ax1, ax2 = fig.add_subplot(2,1,1), fig.add_subplot(2,1,2)

            for model_id in range(len(model_list)):
                pd_mses = pd.DataFrame({"Val": mses[area_id,:,model_id]}, index=pd.Index(pd.date_range(test_period_list[0][0:4]+"/"+test_period_list[0][4:6]+"/01", periods=len(test_period_list), freq='MS'), name="Data"))
                pd_r2s = pd.DataFrame({"Val": r2s[area_id,:,model_id]}, index=pd.Index(pd.date_range(test_period_list[0][0:4]+"/"+test_period_list[0][4:6]+"/01", periods=len(test_period_list), freq='MS'), name="Data"))

                for fig_id in range(4):
                    if fig_id == 0: # mse
                        ax1.plot(pd_mses, label=model_list[model_id], linewidth=2)
                        for rain_id in range(len(rain_start_year_list)):
                            ax1.axvspan(rain_start_datetime[rain_id], rain_end_datetime[rain_id], color="gray", alpha=0.3)
                        if not model_id==0:
                            ax1.legend(loc="upper right")
                            ax1.set_title("Monthly averaged mse")
                            ax1.grid()
                    elif fig_id == 1: # r2
                        ax2.plot(pd_r2s, label=model_list[model_id], linewidth=2)
                        for rain_id in range(len(rain_start_year_list)):
                            ax2.axvspan(rain_start_datetime[rain_id], rain_end_datetime[rain_id], color="gray", alpha=0.3)
                        if not model_id==0:
                            ax2.legend(loc="upper right")
                            ax2.set_title("Monthly averaged r2")
                            ax2.grid()
                    plt.gcf().autofmt_xdate()
            fig.tight_layout()
            fig.savefig(os.path.join(RESULT_DIR, area_list[area_id]+"_result_regression.jpg"), dpi=300)
            fig.clf()


# drow TP/TN/FP/FN plot for each area
def convert_classification_results(x):
    if x.true == 1 and x.pred == 1:
        return "TP" # True Postive
    elif x.true == 1 and x.pred == 0:
        return "FN" # False Negative
    elif x.true == 0 and x.pred == 1:
        return "FP" # False Positive
    else:
        return "TN" # True Negative

# drow difference for each area
def convert_regression_results(x):
    diff = np.abs(x.true - x.pred)
    return diff

def map_visualize_offline_analysis(cfg):

    #################
    # unpack config #
    #################
    DATA_DIR = cfg.DATA_DIR
    RESULT_DIR = cfg.RESULT_DIR
    area_list = cfg.area_list
    model_list = cfg.model_list
    test_period_list = cfg.test_period_list
    eval_period_list = cfg.eval_period_list
    PROBLEM = cfg.PROBLEM

    period_info = cfg.period_info
    mesh_ids = cfg.mesh_ids

    ################
    # load results #
    ################
    npz = np.load(os.path.join(RESULT_DIR, 'results_'+PROBLEM+'.npz'))
    preds, trues = npz['arr_0'], npz['arr_1']

    #######################################################
    # check change points of mesh ids over multiple areas #
    #######################################################
    change_points = get_change_points(mesh_ids)
    #print("change_points: ", change_points)

    ######################################################
    # make comparison maps for each area and each period #
    ######################################################
    for area_id in range(len(area_list)):
        # set mesh geojson information
        if PROBLEM=='classification':
            gdf_mesh_info = gpd.read_file(os.path.join(DATA_DIR, area_list[area_id], "label", "1km_mesh.geojson"))
        elif PROBLEM=='regression':
            gdf_mesh_info = gpd.read_file(os.path.join(DATA_DIR, area_list[area_id], "label", "1km_mesh_deforestation_square_meter.geojson"))
        gdf_mesh_info = gdf_mesh_info.to_crs(4326)

        for test_period_id in range(len(test_period_list)):
            for model_id in range(len(model_list)):
                MODEL_NAME = model_list[model_id]
                #print('model: ', MODEL_NAME)

                # extract preds and trues of the target area (skip mesh_ids)
                area_preds = preds[change_points[area_id]:change_points[area_id+1],test_period_id+1,model_id]
                area_trues = trues[change_points[area_id]:change_points[area_id+1],test_period_id+1,model_id]

                # change data type to pandas DataFrame
                df_area_preds = pd.DataFrame(area_preds)
                df_area_preds = df_area_preds.set_axis(['pred'], axis=1)
                #df_area_preds = df_area_preds.reset_index()

                df_area_trues = pd.DataFrame(area_trues)
                df_area_trues = df_area_trues.set_axis(['true'], axis=1)
                #df_area_trues = df_area_trues.reset_index()

                # make GeoDataFrame
                gdf_area = gpd.GeoDataFrame(pd.concat([df_area_trues,df_area_preds,gdf_mesh_info], axis=1),crs=gdf_mesh_info.crs)

                # make comparison maps
                if PROBLEM=='classification':
                    gdf_area['result'] = gdf_area.apply(convert_classification_results, axis=1)
                    palette = {'TP': 'blue','FP': 'red','FN': 'green','TN': 'grey'}
                    gdf_area['Colors'] = gdf_area['result'].map(palette)

                    custom_points = [Line2D([0], [0], marker="s", linestyle="none", markersize=10, color=color) for color in palette.values()]
                    ax = gdf_area.plot(color=gdf_area['Colors'])
                    ax.legend(custom_points, palette.keys(), bbox_to_anchor=(1, 1), loc='upper left')
                if PROBLEM=='regression':
                    gdf_area['result'] = gdf_area.apply(convert_regression_results, axis=1)
                    ax = gdf_area.plot(column='result', legend=True, legend_kwds={"label": "Absolute difference between true and predicted"},
                                       vmin=0, vmax=1000*1000)

                plt.subplots_adjust(right=0.8)
                plt.title(area_list[area_id]+"_"+ test_period_list[test_period_id])
                plt.tight_layout()
                plt.savefig(os.path.join(RESULT_DIR, area_list[area_id]+"_"+test_period_list[test_period_id]+"_"+MODEL_NAME+"_"+PROBLEM+".png"), dpi=400, facecolor="white", bbox_inches='tight') 
                plt.close()

                # save gdf_area as geojson file
                gdf_area.to_file(driver='GeoJSON', filename=os.path.join(RESULT_DIR, area_list[area_id]+"_"+test_period_list[test_period_id]+"_"+MODEL_NAME+"_"+PROBLEM+".geojson"))