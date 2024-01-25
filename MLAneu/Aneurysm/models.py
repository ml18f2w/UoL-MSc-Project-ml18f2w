import torch.nn as nn


class LogisticRegression(nn.Module):
    '''
    bulid LogisticRegression model
    '''
    def __init__(self, n_features):
        super().__init__()
        self.Linear = nn.Linear(n_features, 1)
        self.sigmoid = nn.Sigmoid()
        self.init_params()

    '''
    initial parameters
    '''
    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                nn.init.constant_(m.bias.data, 0.)

    def forward(self, x):
        return self.sigmoid(self.Linear(x))


class SVM(nn.Module):
    '''
        bulid SVM model
    '''
    def __init__(self, n_features):
        super().__init__()
        self.Linear = nn.Linear(n_features, 1)
        self.init_params()

    '''
        initial parameters
    '''
    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                nn.init.constant_(m.bias.data, 0.)

    def forward(self, x):
        return self.Linear(x)
