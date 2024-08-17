import torch

class LSTM_classification(torch.nn.Module):
    def __init__(self, INPUT_DIM, HIDDEN_DIMS, OUTPUT_DIM, DROPOUT_RATE):
        super().__init__()

        self.lstm1 = torch.nn.LSTM(input_size=INPUT_DIM, hidden_size=HIDDEN_DIMS[0], batch_first=True)
        torch.nn.init.xavier_normal_(self.lstm1.weight_ih_l0)
        torch.nn.init.orthogonal_(self.lstm1.weight_hh_l0)

        self.lstm2 = torch.nn.LSTM(input_size=HIDDEN_DIMS[0], hidden_size=HIDDEN_DIMS[1], batch_first=True)
        torch.nn.init.xavier_normal_(self.lstm2.weight_ih_l0)
        torch.nn.init.orthogonal_(self.lstm2.weight_hh_l0)
        
        self.fc1 = torch.nn.Linear(HIDDEN_DIMS[1], HIDDEN_DIMS[2])
        torch.nn.init.kaiming_normal_(self.fc1.weight)
        self.relu1 = torch.nn.ReLU()
        self.drop1 = torch.nn.Dropout(DROPOUT_RATE)

        self.fc2 = torch.nn.Linear(HIDDEN_DIMS[2], HIDDEN_DIMS[3])
        torch.nn.init.kaiming_normal_(self.fc2.weight)
        self.relu2 = torch.nn.ReLU()
        self.drop2 = torch.nn.Dropout(DROPOUT_RATE)
        
        self.fc_last = torch.nn.Linear(HIDDEN_DIMS[-1], OUTPUT_DIM)
        torch.nn.init.kaiming_normal_(self.fc_last.weight)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x, h = self.lstm1(x)
        x, h = self.lstm2(x)
        x = x[:, -1, :]
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.drop2(x)
        x = self.fc_last(x)
        x = self.softmax(x)

        return x

class LSTM_regression(torch.nn.Module):
    def __init__(self, INPUT_DIM, HIDDEN_DIMS, OUTPUT_DIM, DROPOUT_RATE):
        super().__init__()

        self.lstm1 = torch.nn.LSTM(input_size=INPUT_DIM, hidden_size=HIDDEN_DIMS[0], batch_first=True)
        torch.nn.init.xavier_normal_(self.lstm1.weight_ih_l0)
        torch.nn.init.orthogonal_(self.lstm1.weight_hh_l0)

        self.lstm2 = torch.nn.LSTM(input_size=HIDDEN_DIMS[0], hidden_size=HIDDEN_DIMS[1], batch_first=True)
        torch.nn.init.xavier_normal_(self.lstm2.weight_ih_l0)
        torch.nn.init.orthogonal_(self.lstm2.weight_hh_l0)
        
        self.fc1 = torch.nn.Linear(HIDDEN_DIMS[1], HIDDEN_DIMS[2])
        torch.nn.init.kaiming_normal_(self.fc1.weight)
        self.relu1 = torch.nn.ReLU()
        self.drop1 = torch.nn.Dropout(DROPOUT_RATE)

        self.fc2 = torch.nn.Linear(HIDDEN_DIMS[2], HIDDEN_DIMS[3])
        torch.nn.init.kaiming_normal_(self.fc2.weight)
        self.relu2 = torch.nn.ReLU()
        self.drop2 = torch.nn.Dropout(DROPOUT_RATE)
        
        self.fc_last = torch.nn.Linear(HIDDEN_DIMS[-1], OUTPUT_DIM)
        torch.nn.init.kaiming_normal_(self.fc_last.weight)

    def forward(self, x):
        x, h = self.lstm1(x)
        x, h = self.lstm2(x)
        x = x[:, -1, :]
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.drop2(x)
        x = self.fc_last(x)

        return x


# for hyperparameter search based on skorch
class LSTM_skorch_classification(torch.nn.Module):
    def __init__(self, input_dim, unit_num1, unit_num2, unit_num3, unit_num4, output_dim, dropout_rate):
        super().__init__()

        self.lstm1 = torch.nn.LSTM(input_size=input_dim, hidden_size=unit_num1, batch_first=True)
        torch.nn.init.xavier_normal_(self.lstm1.weight_ih_l0)
        torch.nn.init.orthogonal_(self.lstm1.weight_hh_l0)

        self.lstm2 = torch.nn.LSTM(input_size=unit_num1, hidden_size=unit_num2, batch_first=True)
        torch.nn.init.xavier_normal_(self.lstm2.weight_ih_l0)
        torch.nn.init.orthogonal_(self.lstm2.weight_hh_l0)
        
        self.fc1 = torch.nn.Linear(unit_num2, unit_num3)
        torch.nn.init.kaiming_normal_(self.fc1.weight)
        self.relu1 = torch.nn.ReLU()
        self.drop1 = torch.nn.Dropout(dropout_rate)

        self.fc2 = torch.nn.Linear(unit_num3, unit_num4)
        torch.nn.init.kaiming_normal_(self.fc2.weight)
        self.relu2 = torch.nn.ReLU()
        self.drop2 = torch.nn.Dropout(dropout_rate)
        
        self.fc_last = torch.nn.Linear(unit_num4, output_dim)
        torch.nn.init.kaiming_normal_(self.fc_last.weight)
        self.softmax = torch.nn.Softmax(dim=1)
        
    def forward(self, x):
        x, h = self.lstm1(x)
        x, h = self.lstm2(x)
        x = x[:, -1, :]
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.drop2(x)
        x = self.fc_last(x)
        x = self.softmax(x)

        return x

class LSTM_skorch_regression(torch.nn.Module):
    def __init__(self, input_dim, unit_num1, unit_num2, unit_num3, unit_num4, output_dim, dropout_rate):
        super().__init__()

        self.lstm1 = torch.nn.LSTM(input_size=input_dim, hidden_size=unit_num1, batch_first=True)
        torch.nn.init.xavier_normal_(self.lstm1.weight_ih_l0)
        torch.nn.init.orthogonal_(self.lstm1.weight_hh_l0)

        self.lstm2 = torch.nn.LSTM(input_size=unit_num1, hidden_size=unit_num2, batch_first=True)
        torch.nn.init.xavier_normal_(self.lstm2.weight_ih_l0)
        torch.nn.init.orthogonal_(self.lstm2.weight_hh_l0)
        
        self.fc1 = torch.nn.Linear(unit_num2, unit_num3)
        torch.nn.init.kaiming_normal_(self.fc1.weight)
        self.relu1 = torch.nn.ReLU()
        self.drop1 = torch.nn.Dropout(dropout_rate)

        self.fc2 = torch.nn.Linear(unit_num3, unit_num4)
        torch.nn.init.kaiming_normal_(self.fc2.weight)
        self.relu2 = torch.nn.ReLU()
        self.drop2 = torch.nn.Dropout(dropout_rate)
        
        self.fc_last = torch.nn.Linear(unit_num4, output_dim)
        torch.nn.init.kaiming_normal_(self.fc_last.weight)

    def forward(self, x):
        x, h = self.lstm1(x)
        x, h = self.lstm2(x)
        x = x[:, -1, :]
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.drop2(x)
        x = self.fc_last(x)

        return x