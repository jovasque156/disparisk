import os
import pandas as pd
import numpy as np
from numpy import unique as np_unique
from sklearn.metrics import f1_score, matthews_corrcoef, confusion_matrix

def CI(sensitives):
    """
    returning the class imbalance metric
    assumptions:
        - values in sensitives are eithr 1 or 0
    
    sensitives: array, it assumes 1 is disadv
    """
    return ((sensitives==0).sum()-(sensitives==1).sum())/sensitives.shape[0]

def DPL(labels, sensitives):
    """
    Difference in positive proportions in observed labels

    labels: array, it assumes binary classifications with values 0 and 1, and 1 as the positive class
    sensitives: array, it assumes 1 is disadv
    """
    n_a = labels[sensitives==0]
    n_d = labels[sensitives==1]

    q_a = n_a.sum()/n_a.shape[0]
    q_d = n_d.sum()/n_d.shape[0]

    return q_a - q_d

def KL(labels, sensitives):
    """
    KL divergence between positive and negative distributions

    labels: array, it assumes binary classifications with values 0 and 1, and 1 as the positive class
    sensitives: array, it assumes 1 is disadv
    """
    n_d = labels[sensitives==1]
    n_a = labels[sensitives==0]

    q_a = n_a.sum()/n_a.shape[0]
    q_d = n_d.sum()/n_d.shape[0]

    return q_a*np.log(q_a/q_d)+(1-q_a)*np.log((1-q_a)/(1-q_d))

def phi_coef(labels, sensitives):
    """
    Matthews Corr coef coeficiente 
    labels: array, assuming values in {0,1}
    predictions: array, assuming values in {0,1}
    """

    return matthews_corrcoef(labels, sensitives)

def phi_coef_own(labels, sensitives):
    """
    Matthews Corr coef coeficiente 
    labels: array, assuming values in {0,1}
    predictions: array, assuming values in {0,1}
    """
    cm = confusion_matrix(labels, sensitives)

    num = (cm[1,1]*cm[0,0]-cm[1,0]*cm[0,1])
    dem = (np.sqrt(cm[1,:].sum())*np.sqrt(cm[0,:].sum())*np.sqrt(cm[:,0].sum())*np.sqrt(cm[:,1].sum()))

    return num/dem


def f1score(labels, predictions, ave='binary'):
    """
    Returning f1-score
    """
    return f1_score(y_true=labels, y_pred=predictions, average=ave)

def accuracy(labels, predictions):
    """
    return the accuracy given labels and predictions
    labels: array
    predictions: array
    """
    return (labels == predictions).sum() / labels.shape[0]

def dem_p(predictions, sensitives):
    """
    returns the demographic parity given predictions and sensitives values
    
    predictions: array, it assumes binary classifications with values 0 and 1
    sensitives: array, it assumes 1 is disadv
    """
    avg_positve_rate = (predictions==1).sum()/predictions.shape[0]

    sign = 1

    total = 0
    for group in np_unique(sensitives):
        filter_pred = predictions[sensitives == group]
        dif = avg_positve_rate - ((filter_pred == 1).sum()/filter_pred.shape[0])
        
        if group==1:
            if dif>=0:
                sign = 1
            else:
                sign = -1

        total += abs(dif)
    
    return sign*(total)

def eq_opp(predictions, labels, sensitives):
    """
    returns the equalized opportunity given predictions, labels, and sensitives

    predictions: array, it assumes binary classifications with values 0 and 1
    labels: array, it assumes binary classifications with values 0 and 1
    sensitives, array
    """
    positive_pred = predictions[labels == 1]
    positive_sensitives = sensitives[labels == 1]
    ave_recall = (positive_pred == 1).sum() / positive_pred.shape[0]
    sign = 1
    total = 0
    
    for group in np_unique(positive_sensitives):
        filter_pred = positive_pred[positive_sensitives == group]
        dif = ave_recall - ((filter_pred == 1).sum() / filter_pred.shape[0])
        
        if group==1:
            if dif >= 0:
                sign = 1
            else:
                sign = -1
        
        total += abs(dif)
    return sign*(total)

def predictive_equality(predictions, labels, sensitives):
    """
    returns the predictive equality given predictions, labels, and sensitives

    predictions: array, it assumes binary classifications with values 0 and 1
    labels: array, it assumes binary classifications with values 0 and 1
    sensitives, array
    """
    neg_pred = predictions[labels != 1]
    neg_sensitives = sensitives[labels != 1]
    ave_recall = (neg_pred == 1).sum() / neg_pred.shape[0]
    sign = 1
    total = 0
    
    for group in np_unique(neg_sensitives):
        filter_pred = neg_pred[neg_sensitives == group]
        dif = ave_recall - ((filter_pred == 1).sum() / filter_pred.shape[0])
        
        if group==1:
            if dif > 0:
                sign = -1
            else:
                sign = 1
        
        total += abs(dif)
    return sign*(total)

def eq_odd(predictions, labels, sensitives):
    """
    returns the equalized opportunity given predictions, labels, and sensitives

    predictions: array, it assumes binary classifications with values 0 and 1
    labels: array, it assumes binary classifications with values 0 and 1
    sensitives, array
    """
    negative_pred = predictions[labels == 0]
    negative_sensitives = sensitives[labels == 0]
    ave_fpr = (negative_pred == 1).sum() / negative_pred.shape[0]
    max_val = 0
    min_val = 1
    for group in np_unique(negative_pred):
        filter_pred = negative_pred[negative_sensitives == group]
        dif = ave_fpr - (filter_pred == 1).sum() / filter_pred.shape[0]
        if max_val < dif:
            max_val = dif
        if min_val > dif:
            min_val = dif

    return eq_opp(predictions==0, labels==0, sensitives) / 2 + eq_opp(predictions, labels, sensitives) / 2

def accuracy_dif(predictions, labels, sensitives):
    """
    returns the accuracy difference given predictions, labels, and sensitives

    predictions: array, it can be multiclass
    labels: array, it assumes binary classifications with values 0 and 1
    sensitives, array
    """
    acc_disadv = accuracy(labels[sensitives==1], predictions[sensitives==1])
    acc_adv = accuracy(labels[sensitives==0], predictions[sensitives==0])

    return acc_adv - acc_disadv

def obtain_pvis(base_path_to_pvis, family_representatives, dict_label, label, sensitive, disadv, dict_label_hs=None, class_label=None,sensitive_too = False, use_equivelance=None, set_cal='val', verbose=False):
    """
    Compute the pvis
    values in dict_label must be the same in the label column of pvis
    disadv value must appear in the sensitive
    """
    columns = {}

    columns["family"] = []
    columns["epoch"] = []

    variables = [label] if not sensitive_too else [label, sensitive]

    for set_ in ['val', 'test']:
        for var in variables:
            columns[f"Ev_{var}_{set_}"] = []
            
            if var == label:
                columns[f"Ev_{label}_{set_}_adv"] = []
                columns[f"Ev_{label}_{set_}_disadv"] = []
                
                if class_label:
                    class_=class_label
                    columns[f"Ev_{label}_{set_}_{class_}"] = []
                    columns[f"Ev_{label}_{set_}_{class_}_adv"] = []
                    columns[f"Ev_{label}_{set_}_{class_}_disadv"] = []
                    
                    columns[f"Ev_{label}_{set_}_not{class_}"] = []
                    columns[f"Ev_{label}_{set_}_not{class_}_adv"] = []
                    columns[f"Ev_{label}_{set_}_not{class_}_disadv"] = []
                else:
                    if dict_label_hs:
                        for class_ in dict_label_hs:
                            columns[f"Ev_{label}_{set_}_{class_}"] = []
                            columns[f"Ev_{label}_{set_}_{class_}_adv"] = []
                            columns[f"Ev_{label}_{set_}_{class_}_disadv"] = []
                            columns[f"porc_{label}_{set_}_{class_}"] = []
                    else:
                        for class_ in dict_label:
                            columns[f"Ev_{label}_{set_}_{class_}"] = []
                            columns[f"Ev_{label}_{set_}_{class_}_adv"] = []
                            columns[f"Ev_{label}_{set_}_{class_}_disadv"] = []
                            columns[f"porc_{label}_{set_}_{class_}"] = []
                

    for family in family_representatives:
        for epoch in range(1,6):
            columns["family"] += [family]
            columns["epoch"] += [epoch]
            
            for var in variables:
                for set_ in ['val', 'test']:
                    if os.path.isfile(os.path.join(base_path_to_pvis, family_representatives[family], f"{var}_std", f"pvis_on_{set_}_calibrator_{set_cal}.csv")):
                        if verbose: print(f'using calibrator at {family}')
                        path = os.path.join(base_path_to_pvis, family_representatives[family], f"{var}_std", f"pvis_on_{set_}_calibrator_{set_cal}.csv")
                    else:
                        if class_label:
                            if verbose: print(f'using regular at {family}, {class_label}')
                            path = os.path.join(base_path_to_pvis, family_representatives[family], f"{var}_{class_label}_std", f"pvis_on_{set_}.csv")
                        else:
                            if verbose: print(f'using regular at {family}')
                            path = os.path.join(base_path_to_pvis, family_representatives[family], f"{var}_std", f"pvis_on_{set_}.csv")
                    pvis = pd.read_csv(path)

                    Ev_null = pvis[f"Hy_null_{var}_epoch1"].mean()
                    Ev = pvis[f"Hy_std_{var}_epoch{epoch}"].mean()
                    columns[f"Ev_{var}_{set_}"] += [Ev]
                    columns[f"Ev_{var}_{set_}_null"] += [Ev_null]

                    if var==label:
                        labels = pvis[f"label_{label}"].values
                        labels_list = list(labels)
                        if len([k for k in dict_label.keys() if k in labels_list])>0:
                            labels = np.array([dict_label[l] for l in pvis[f'label_{label}'].values])
                        
                        Ev_adv = pvis[pvis[f"sensitive_{sensitive}"]!=disadv][f"Hy_std_{var}_epoch{epoch}"].mean()
                        
                        Ev_disadv = pvis[pvis[f"sensitive_{sensitive}"]==disadv][f"Hy_std_{var}_epoch{epoch}"].mean()
                        
                        columns[f"Ev_{var}_{set_}_adv"] += [Ev_adv]
                        columns[f"Ev_{var}_{set_}_disadv"] += [Ev_disadv]
                        
                        path_class = os.path.join(base_path_to_pvis, family_representatives[family], f"{var}_std", f"pvis_on_{set_}_epoch{epoch}.csv")
                        if os.path.isfile(path_class):
                            pvis_class = pd.read_csv(path_class)
                        else:
                            pvis_class = None
                        
                        if class_label:
                            class_ = class_label
                            e_class = pvis[(labels==dict_label[class_])][f"Hy_std_{label}_epoch{epoch}"].mean()
                            columns[f"Ev_{label}_{set_}_{class_}"] += [e_class]
                            
                            e_class_disadv = pvis[(pvis[f"sensitive_{sensitive}"]==disadv) & (labels==dict_label[class_])][f"Hy_std_{label}_epoch{epoch}"].mean()
                            columns[f"Ev_{label}_{set_}_{class_}_disadv"] += [e_class_disadv]
                            
                            e_class_adv = pvis[(pvis[f"sensitive_{sensitive}"]!=disadv) & (labels==dict_label[class_])][f"Hy_std_{label}_epoch{epoch}"].mean()
                            columns[f"Ev_{label}_{set_}_{class_}_adv"] += [e_class_adv]
                            
                            e_class = pvis[(labels!=dict_label[class_])][f"Hy_std_{label}_epoch{epoch}"].mean()
                            columns[f"Ev_{label}_{set_}_not{class_}"] += [e_class]
                            
                            e_class_disadv = pvis[(pvis[f"sensitive_{sensitive}"]==disadv) & (labels!=dict_label[class_])][f"Hy_std_{label}_epoch{epoch}"].mean()
                            columns[f"Ev_{label}_{set_}_not{class_}_disadv"] += [e_class_disadv]
                            
                            e_class_adv = pvis[(pvis[f"sensitive_{sensitive}"]!=disadv) & (labels!=dict_label[class_])][f"Hy_std_{label}_epoch{epoch}"].mean()
                            columns[f"Ev_{label}_{set_}_not{class_}_adv"] += [e_class_adv]
                        else:
                            if dict_label_hs:
                                dict_label_final = dict_label_hs
                            else:
                                dict_label_final = dict_label

                            for class_ in dict_label_final:
                                if class_ == 'harrasement':
                                    porc = sum(labels<2)/len(pvis)
                                else:
                                    porc = sum(labels==dict_label[class_])/len(pvis)
                                columns[f"porc_{label}_{set_}_{class_}"] += [porc]

                                if pvis_class:
                                    e_class = pvis_class[f"Hy_{dict_label[class_]}_std_{label}_epoch{epoch}"].mean()
                                    columns[f"Ev_{label}_{set_}_{class_}"] += [e_class]
                                    
                                    e_class_disadv = pvis[(pvis[f"sensitive_{sensitive}"]==disadv)][f"Hy_{dict_label[class_]}_std_{label}_epoch{epoch}"].mean()
                                    columns[f"Ev_{label}_{set_}_{class_}_disadv"] += [e_class_disadv]
                                    
                                    e_class_adv = pvis[(pvis[f"sensitive_{sensitive}"]!=disadv)][f"Hy_{dict_label[class_]}_std_{label}_epoch{epoch}"].mean()
                                    columns[f"Ev_{label}_{set_}_{class_}_adv"] += [e_class_adv]
                                    
                                else:
                                    if class_ == 'harrasement':
                                        e_class = pvis[(labels<2)][f"Hy_std_{label}_epoch{epoch}"].mean()
                                        columns[f"Ev_{label}_{set_}_{class_}"] += [e_class]
                                        
                                        e_class_disadv = pvis[(pvis[f"sensitive_{sensitive}"]==disadv) & (labels<2)][f"Hy_std_{label}_epoch{epoch}"].mean()
                                        columns[f"Ev_{label}_{set_}_{class_}_disadv"] += [e_class_disadv]
                                        
                                        e_class_adv = pvis[(pvis[f"sensitive_{sensitive}"]!=disadv) & (labels<2)][f"Hy_std_{label}_epoch{epoch}"].mean()
                                        columns[f"Ev_{label}_{set_}_{class_}_adv"] += [e_class_adv]
                                    else:
                                        e_class = pvis[(labels==dict_label[class_])][f"Hy_std_{label}_epoch{epoch}"].mean()
                                        columns[f"Ev_{label}_{set_}_{class_}"] += [e_class]
                                        
                                        e_class_disadv = pvis[(pvis[f"sensitive_{sensitive}"]==disadv) & (labels==dict_label[class_])][f"Hy_std_{label}_epoch{epoch}"].mean()
                                        columns[f"Ev_{label}_{set_}_{class_}_disadv"] += [e_class_disadv]
                                        
                                        e_class_adv = pvis[(pvis[f"sensitive_{sensitive}"]!=disadv) & (labels==dict_label[class_])][f"Hy_std_{label}_epoch{epoch}"].mean()
                                        columns[f"Ev_{label}_{set_}_{class_}_adv"] += [e_class_adv]

    return pd.DataFrame(columns)

def obtain_perfo(base_path_to_pvis, models_per_family, dict_label, label, sensitive, disadv, dict_label_hs=None, class_label=None, sensitive_too = False, set_cal='val', verbose = False):
    columns = {}

    columns["family"] = []
    columns["model"] = []
    columns["epoch"] = []

    for set_ in ['val', 'test']:
        columns[f"accuracy_{set_}"] = []
        columns[f"f1_{set_}"] = []
        columns[f"loss_{set_}"] = []
        
        if class_label:
            class_ = class_label
            columns[f"porc_{set_}_{class_}"] = []
            columns[f"demp_{set_}_{class_}"] = []
            columns[f"eqopp_{set_}_{class_}"] = []
            columns[f"eqodd_{set_}_{class_}"] = []
            columns[f"accuracy_{set_}_{class_}"] = []
        else:
            if dict_label_hs:
                for class_ in dict_label_hs:
                    columns[f"porc_{set_}_{class_}"] = []
                    columns[f"demp_{set_}_{class_}"] = []
                    columns[f"eqopp_{set_}_{class_}"] = []
                    columns[f"eqodd_{set_}_{class_}"] = []
                    columns[f"accuracy_{set_}_{class_}"] = []
                    columns[f"f1_{set_}_{class_}"] = []
            else:
                for class_ in dict_label:
                    columns[f"porc_{set_}_{class_}"] = []
                    columns[f"demp_{set_}_{class_}"] = []
                    columns[f"eqopp_{set_}_{class_}"] = []
                    columns[f"eqodd_{set_}_{class_}"] = []
                    columns[f"accuracy_{set_}_{class_}"] = []
                    columns[f"f1_{set_}_{class_}"] = []

    for family in models_per_family:
        for model in models_per_family[family]:
            for epoch in range(1,6):
                columns["family"] += [family]
                columns["model"] += [model]
                columns["epoch"] += [epoch]
                
                for set_ in ['val', 'test']:
                    if os.path.isfile(os.path.join(base_path_to_pvis, model, f"{label}_std", f"predictions_on_{set_}.csv")):
                        path = os.path.join(base_path_to_pvis, model, f"{label}_std", f"predictions_on_{set_}.csv")
                    else:
                        if os.path.isfile(os.path.join(base_path_to_pvis, model, f"{label}_std", f"pvis_on_{set_}_calibrator_{set_cal}.csv")):
                            if verbose: print(f'using calibrator at {family}')
                            path = os.path.join(base_path_to_pvis, model, f"{label}_std", f"pvis_on_{set_}_calibrator_{set_cal}.csv")
                        else:
                            if class_label:
                                if verbose: print(f'using regular at {family}, {class_label}')
                                path = os.path.join(base_path_to_pvis, model, f"{label}_{class_label}_std", f"pvis_on_{set_}.csv")
                            else:
                                if verbose: print(f'using regular at {family}')
                                path = os.path.join(base_path_to_pvis, model, f"{label}_std", f"pvis_on_{set_}.csv")
                                        
                    metrics = pd.read_csv(path)

                    #obtain pvis
                    if os.path.isfile(os.path.join(base_path_to_pvis, model, f"{label}_std", f"pvis_on_{set_}_calibrator_{set_cal}.csv")):
                        if verbose: print(f'using calibrator at {family}')
                        path = os.path.join(base_path_to_pvis, model, f"{label}_std", f"pvis_on_{set_}_calibrator_{set_cal}.csv")
                    else:
                        if class_label:
                            if verbose: print(f'using regular at {family}, {class_label}')
                            path = os.path.join(base_path_to_pvis, model, f"{label}_{class_label}_std", f"pvis_on_{set_}.csv")
                        else:
                            if verbose: print(f'using regular at {family}')
                            path = os.path.join(base_path_to_pvis, model, f"{label}_std", f"pvis_on_{set_}.csv")
                    pvis = pd.read_csv(path)

                    predictions = metrics[f'prediction_{label}_epoch{epoch}'].values
                    labels = metrics[f'label_{label}'].values
                    labels_list = list(labels)

                    if len([k for k in dict_label.keys() if k in labels_list])>0:
                        labels = np.array([dict_label[l] for l in metrics[f'label_{label}'].values])
                    if len(set(labels_list))>0 and class_label:
                        labels = np.array([1 if l==dict_label[class_label] else 0 for l in labels])
                    sensitives = np.array([int(s==disadv) for s in metrics[f'sensitive_{sensitive}'].values])

                    columns[f"accuracy_{set_}"] += [accuracy(labels, predictions)]
                    columns[f"f1_{set_}"] += [f1score(labels, predictions)] if len(labels_list) < 3 else [f1score(labels, predictions, ave='weighted')]
                    columns[f"loss_{set_}"] += [pvis[f"Hy_std_{label}_epoch{epoch}"].mean()]
                    
                    if class_label:
                        class_ = class_label
                        porc = sum(labels==1)/len(labels)
                        columns[f"porc_{set_}_{class_}"] += [porc]
                        
                        demp = dem_p(predictions, sensitives)
                        columns[f"demp_{set_}_{class_}"] += [demp]
                        
                        eqopp = eq_opp(predictions, labels, sensitives)
                        columns[f"eqopp_{set_}_{class_}"] += [eqopp]
                        
                        eqodd = eq_odd(predictions, labels, sensitives)
                        columns[f"eqodd_{set_}_{class_}"] += [eqodd]

                        acc = accuracy(labels, predictions)
                        columns[f"accuracy_{set_}_{class_}"] += [acc]

                        columns[f"f1_{set_}_{class_}"] += [f1score(labels, predictions)] if len(labels_list) < 3 else [f1score(labels, predictions)]
                    else:
                        if dict_label_hs:
                            dict_label_final = dict_label_hs
                        else:
                            dict_label_final = dict_label
                                
                        for class_ in dict_label_final:
                            if class_=='harrasement':
                                porc = sum(labels<2)/len(labels)
                            else:
                                porc = sum(labels==dict_label[class_])/len(labels)
                            columns[f"porc_{set_}_{class_}"] += [porc]
                            
                            if class_ == 'harrasement':
                                demp = dem_p((predictions<2).astype(int), sensitives)
                                columns[f"demp_{set_}_{class_}"] += [demp]
                                
                                eqopp = eq_opp((predictions<2).astype(int), (labels<2).astype(int), sensitives)
                                columns[f"eqopp_{set_}_{class_}"] += [eqopp]
                                
                                eqodd = eq_odd((predictions<2).astype(int), (labels<2).astype(int), sensitives)
                                columns[f"eqodd_{set_}_{class_}"] += [eqodd]

                                acc = accuracy((labels<2).astype(int), (predictions<2).astype(int))
                                columns[f"accuracy_{set_}_{class_}"] += [acc]

                                columns[f"f1_{set_}_{class_}"] += [f1score(labels, predictions)] if len(labels_list) < 3 else [f1score((labels<2).astype(int), (predictions<2).astype(int))]
                            else:
                                demp = dem_p((predictions==dict_label[class_]).astype(int), sensitives)
                                columns[f"demp_{set_}_{class_}"] += [demp]
                                
                                eqopp = eq_opp((predictions==dict_label[class_]).astype(int), (labels==dict_label[class_]).astype(int), sensitives)
                                columns[f"eqopp_{set_}_{class_}"] += [eqopp]
                                
                                eqodd = eq_odd((predictions==dict_label[class_]).astype(int), (labels==dict_label[class_]).astype(int), sensitives)
                                columns[f"eqodd_{set_}_{class_}"] += [eqodd]

                                acc = accuracy((labels==dict_label[class_]).astype(int), (predictions==dict_label[class_]).astype(int))
                                columns[f"accuracy_{set_}_{class_}"] += [acc]

                                columns[f"f1_{set_}_{class_}"] += [f1score(labels, predictions)] if len(labels_list) < 3 else [f1score((labels==dict_label[class_]).astype(int), (predictions==dict_label[class_]).astype(int))]

    return pd.DataFrame(columns)