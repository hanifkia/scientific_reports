# scientific_reports
An automate pipeline to generate k-fold cross validation reports for classification problems

how to use:
```bash
import sys
sys.path.append('<path to scientific_reports>')
from scientific_reports import scientificReports
```

create dicts:
```bash
model_dict = {
    'XGBoost' : {
        'model' : XGBClassifier,
        'model_params' : {'gamma': 0, 'max_delta_step': 1, 'scale_pos_weight': 110, 'use_label_encoder' : False},
        'fit_params' : {'eval_metric' : 'error'}
    },
    'SVM' : {
        'model' : SVC,
        'model_params' : {'C': 100, 'gamma': 'auto', 'kernel': 'rbf'},
        'fit_params' : {}
    },
    'MLP' : {
        'model' : MLPClassifier,
        'model_params' : {'input_shape':(xData.shape[1],), 'hidden_layer_units' : [128, 64, 64], 'n_classes':len(np.unique(yData)), 'dropout':0.25},
        'fit_params' : {'epochs':100, 'learning_rate':'schedule', 'ballence':True}
    },
}


ordered_scalers_list = [MinMaxScaler()]

metrics_dict = {
    'F2-Score' : {
        'func' : fbeta_score,
        'params' : {'beta':2}
    },
    'Accuracy' : {
        'func' : accuracy_score,
        'params' : {}
    },
    'Roc-Auc' : {
        'func' : roc_auc_score,
        'params' : {}
    },
    'G-Mean' : {
        'func' : geometric_mean_score,
        'params' : {}
    },
    'MCC' : {
        'func' : matthews_corrcoef,
        'params' : {}
    },
}
```
call class:

```bash
scr = scientificReports(model_dict, metrics_dict, ordered_scalers_list, n_splits=5, n_repeats=3, verbose=1)
res = scr.run(xData.values, yData, tab='all', save=True)
res
```

parameters:
```bash
tab: if 'all' reports mean and std of all folds, if 'max' reports highest fold, if int(n) reports mean and std of first n folds.
save: if true save results as csv in reports directory with specific name.
```
