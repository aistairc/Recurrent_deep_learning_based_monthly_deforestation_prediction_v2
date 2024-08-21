import os

class config:
    def __init__(self):
        # directory info
        self.DATA_DIR = os.path.join('.', 'data')
        self.MODEL_DIR = os.path.join('.', 'model')
        self.RESULT_DIR = os.path.join('.', 'result')
        self.PREDICT_DIR = os.path.join('.', 'predict')

        os.makedirs(self.MODEL_DIR, exist_ok=True)
        os.makedirs(self.RESULT_DIR, exist_ok=True)
        os.makedirs(self.PREDICT_DIR, exist_ok=True)

        # target info
        # !! test and eval periods are based on label. For example, if 202110 is the test and eval periods, 
        #    the label is 202110 and the features have information up to 202109.
        self.area_list = ["porto_velho","humaita","altamira",                            # small-scale logging by poor farmers
                          "vista_alegre_do_abuna","novo_progresso","sao_felix_do_xingu", # large-scale logging by large landowners
                          "S6W57","S7W57"]                                               # mining development
        self.feature_list = ["deforestation_event", "deforestation_square_meter",
                             "distance_to_closest_road_meter"]
        self.model_list = ["LSTM", "LR"] # "LR" has different meaning in classification and regression (classification: Logistic Regression, regression: Linear Regression)
        self.entire_period_list = ["201710","201711","201712",
                                   "201801","201802","201803","201804","201805","201806","201807","201808","201809","201810","201811","201812",
                                   "201901","201902","201903","201904","201905","201906","201907","201908","201909","201910","201911","201912",
                                   "202001","202002","202003","202004","202005","202006","202007","202008","202009","202010","202011","202012",
                                   "202101","202102","202103","202104","202105","202106","202107","202108","202109","202110","202111","202112",
                                   "202201","202202","202203","202204","202205","202206","202207","202208","202209"]
        self.train_end_period_list = ["202009","202010","202011","202012",
                                      "202101","202102","202103","202104","202105","202106","202107","202108","202109","202110","202111","202112",
                                      "202201","202202","202203","202204","202205","202206","202207","202208"]
        self.test_period_list = ["202010","202011","202012",
                                 "202101","202102","202103","202104","202105","202106","202107","202108","202109","202110","202111","202112",
                                 "202201","202202","202203","202204","202205","202206","202207","202208","202209"]
        self.eval_period_list = ["202106","202107","202108","202109",
                                 "202206","202207","202208","202209"]

        ######################################
        # you can change target problem here #
        ######################################
        # classification: deforestation_event
        # regression    : deforestation_square_meter
        self.OBJECTIVE_VARIABLE = "deforestation_event"
        #self.OBJECTIVE_VARIABLE = "deforestation_square_meter"
        ######################################

        if self.OBJECTIVE_VARIABLE == "deforestation_event":
            self.PROBLEM = 'classification'
        elif self.OBJECTIVE_VARIABLE == "deforestation_square_meter":
            self.PROBLEM = 'regression'
        else:
            print('error: you should set objective variable to deforestation_event or deforestation_square_meter')

        # dataset info
        self.DURATION = 12
        self.REMAKING = False

        # hyperparam search info
        self.unit_num_range = [16, 32, 64, 128] 
        self.batch_size_range = [64, 128, 256]
        self.lr_range =  [0.0001, 0.001, 0.01, 0.1]
        self.weight_decay_range = [0, 0.0001, 0.001, 0.01]
        self.SEED = 25
        self.CV_NUM = 4
        self.EPOCH_NUM = 100
        self.PATIENCE = 5
        self.DROPOUT_RATE = 0.5
        self.WORKER_NUM = 14 # depends on your PC (recommend: number of CPU cores - 4)
        if self.PROBLEM == 'classification':
            self.SCORE = 'neg_log_loss' # LogLoss
        elif self.PROBLEM == 'regression':
            self.SCORE = 'neg_mean_squared_error' # MSE

        # neural network info
        self.RETRAINING = False

        # evaluation info
        self.REEVALUATION = False
        self.rain_start_year_list = [2020, 2021] # from october
        self.rain_end_year_list = [2021, 2022]   # to march
        self.PROB_THRESHOLD = 0.7
        self.MASKING = True
        self.MASKING_THRESHOLD = 900*900

        # prediction info
        self.FUTURE_PREDICT = True
        self.INPUT_FEATURE_PERIOD_FOR_PREDICTION = "202209"  # predict next period