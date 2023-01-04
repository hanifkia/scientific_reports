import numpy as np
import pandas as pd
import datetime
import copy
import os
from datetime import datetime
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split

class scientificReports():
    def __init__(self, model_dict, metrics_dict, scalers_list = [], n_splits=10, n_repeats=3, verbose=1, desc=''):
        self.model_dict = model_dict
        self.metrics_dict = metrics_dict
        self.scalers_list = scalers_list
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.verbose = verbose
        self.desc = desc
    
    def __scale__(self, xTrain, xTest):
        for s in self.scalers_list:
            xTrain = s.fit_transform(xTrain)
            xTest = s.transform(xTest)
        return xTrain, xTest
        
    def __try__(self, x_train, y_train, x_test ,model):
        step_res = {}            
        m = model['model'](**model['model_params'])
        m.fit(x_train, y_train, **model['fit_params'])
        y_pred = m.predict(x_test)
        return y_pred

    def __pipeline(self, X, Y):
        # define initial result format
        res = {}
        for name in self.model_dict.keys():
            res[name] = {met : [] for met in self.metrics_dict.keys()}
        # generate kfold indexes:   
        rskf = RepeatedStratifiedKFold(n_splits=self.n_splits, n_repeats=self.n_repeats, random_state=1)
        jj = 1
        for train_index, test_index in rskf.split(X, Y):
            if self.verbose == 1:
                print(f'*********** fold {self.n_splits*self.n_repeats} / {jj} ***********')
            elif self.verbose == 2:
                print(f'fold {self.n_splits*self.n_repeats} / {jj}')
            x_train, y_train = X[train_index], Y[train_index]
            x_test, y_test = X[test_index], Y[test_index]        
            
            # scaling
            if len(self.scalers_list) != 0:
                x_train, x_test = self.__scale__(x_train, x_test)
            
            # prediction
            for model in self.model_dict.keys():
                if self.verbose == 1:
                    start_time = datetime.now().timestamp()
                    print('[start] ', model)
                y_pred = self.__try__(x_train, y_train, x_test, self.model_dict[model])
                #performance
                if type(y_pred) == dict:
                    res.pop(model, None)
                    for name in y_pred.keys():
                        if name not in res.keys():
                            res[name] = {met : [] for met in self.metrics_dict.keys()}
                        performance_tuple = (y_test, y_pred[name])
                        for metric in self.metrics_dict.keys():
                            res[name][metric].append(self.metrics_dict[metric](*performance_tuple))                
                else:
                    performance_tuple = (y_test, y_pred)
                    for metric in self.metrics_dict.keys():
                        res[model][metric].append(self.metrics_dict[metric]['func'](*performance_tuple, **self.metrics_dict[metric]['params']))
                        
                if self.verbose == 1:      
                    print('[finish] ', model, f' : learning time: {round(datetime.now().timestamp() - start_time)} (seconds)', '\n')
            jj+=1
        return res
    
    def get_table(self, d):
        res = copy.deepcopy(d)
        for item in res.keys():
            for met in res[item].keys():
                tmp1 = 1 if met == 'MCC' else 100
                tmp2 = 2 if met == 'MCC' else 1
                # +/- : \u00B1
                res[item][met] = f'{round((np.mean(res[item][met]) * tmp1), tmp2)} (se: {round(((np.std(res[item][met]) / (np.sqrt(self.n_splits*self.n_repeats))) * tmp1), tmp2)})'
        return pd.DataFrame(res).T
    
    def saveTab(self, tab, saved_path = './reports/'):
        if os.path.exists(saved_path):
            pass
        else:
            os.mkdir(saved_path)
        s = datetime.now()
        path = saved_path + 'report_' + f'{s.year}-{str(s.month).zfill(2)}-{str(s.day).zfill(2)}_{str(s.hour).zfill(2)}{str(s.minute).zfill(2)}{str(s.second).zfill(2)}'+self.desc+'.csv'
        tab.to_csv(path)
    
    def run(self, X, Y, tab = False, save=False):
        self.res = self.__pipeline(X, Y)
        if tab:
            rep = self.get_table(self.res)
            if save == True:
                self.saveTab(rep)
            return rep