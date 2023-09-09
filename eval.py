import numpy as np

def perf_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)): 
        if y_actual[i]==y_hat[i]==1:
            TP += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
            FP += 1
        if y_actual[i]==y_hat[i]==0:
            TN += 1
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
            FN += 1

    return(TP, FP, TN, FN)

def get_eval_metrics(ys, y_pred):
    TP, FP, TN, FN  = perf_measure(ys, y_pred)

    acc = sum(ys == y_pred)/len(ys)
    sen = np.float64(TP)/(TP + FN)
    spe = np.float64(TN)/(TN + FP)
    ppv = np.float64(TP)/(TP + FP)
    npv = np.float64(TN)/(TN + FN)

    return {'acc': acc, 'sen': sen, 'spe': spe, 'ppv': ppv, 'npv': npv}

def get_cv_metrics(cv_results):
    accs = [split['acc'] for split in cv_results]
    sens = [split['sen'] for split in cv_results]
    specs = [split['spe'] for split in cv_results]

    mean_accs, std_accs = np.mean(accs), np.std(accs)
    mean_sen, std_sen = np.mean(sens), np.std(sens)
    mean_spec, std_spec = np.mean(specs), np.std(specs)

    return {'acc_mean': mean_accs, 'acc_std': std_accs, 
            'sen_mean': mean_sen, 'sen_std': std_sen,
            'spec_mean': mean_spec, 'spec_std': std_spec}