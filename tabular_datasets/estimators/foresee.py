import numpy as np
import random
import source.fairness as fm
import source.pipes as op
from sklearn.tree import DecisionTreeClassifier

class FORESEE(object):
    def __init__(self, 
                criterion=['entropy'],
                splitter=['random'],
                max_depth=[None], 
                min_samples_leaf=[2], 
                class_weight=['balanced', None], 
                verb = True):

        self.parameters = {'DT__criterion': criterion,
                            'DT__splitter': splitter,
                            'DT__max_depth': max_depth,
                            'DT__min_samples_leaf': min_samples_leaf}
        
        self.class_weight = class_weight
        
        #Variables for computing risk
        self.k_dt = {}      
        self.performances = []
        self.levels_discr = {}
        self.leaves = {}
        self.verb = verb
        
    def fit(self, X, A, unpriv_class, y, ratio_sub_sample= 0.3, number_dim_rem = 1,  k=200, scoring =['f1']):
        '''
        Fit the estimator on the training dataset X and target variable Y.

        Inputs:
        X: numpy (n,m), representing the n samples with m features
        y: numpy (n,), representing the target variable
        A: numpy (n,), representing the sensitive attribute of n samples.
        unpriv_class: int, representing the unprivileged class.
        k: int, representing the number of DT.
        ratio_sub_sample: float, representing the ratio to sampling from X for each DT.
        number_dim_rem: int, representing the number of dimensions to remove from X for each DT.
        scoring: list of str, representing the scoring metrics to use in GridSearchCV.
        '''
        
        if self.verb: print(f'Fitting {k} DT...', end='\r') 
        self.__fit_dts(X, y, ratio_sub_sample, number_dim_rem,  k, scoring)
        if self.verb: print('Computing foresee basics...', end='')
        self.__foresee_basics(X, y, A, unpriv_class)
        if self.verb: print("Complete")
        
        
    def __fit_dts(self, X, y, ratio_sub_sample, number_dim_rem, k, scoring):
        '''
        Fit the set of k of decision-tree-classifiers on sub samples of training dataset X and 
        target variable Y. For each decision-tree saves a 3-tuple of the GridSearchCV with the tunned DT,
        an array of the index of sub-sample, and an array with the sub-sample features.
        
        By default, each DT is trained in a sub_sample of 30% of X and with 1 dimension less.
        
        Inputs:
        X: numpy (n,m), representing the n samples with m features
        y: numpy (n,), representing the target variable
        k: int, representing the number of DT.
        ratio_sub_sample: float, representing the ratio to sampling from X for each DT.
        number_dim_rem: int, representing the number of dimensions to randomly remove for each DT. 
        '''
        
        dt = DecisionTreeClassifier(random_state=0)

        for i in range(0,k):
            if i > int(k/2):
                self.parameters['DT__class_weight']=[self.class_weight[0]]
            else:
                self.parameters['DT__class_weight']=[self.class_weight[1]]
            
            sel = np.array(random.choices(range(X.shape[0]), k = int(X.shape[0]*ratio_sub_sample)))
            sel_var = np.array(random.sample(range(X.shape[1]), k = int(X.shape[1]-number_dim_rem)))
            sel.sort()
            sel_var.sort()
            X_sample = X[sel,:][:,sel_var]
            y_sample = y[sel]

            fit = op.get_grid(X_sample, y_sample, self.parameters, dt, 'DT', scoring= scoring, refit=scoring[0])

            self.k_dt["DT_"+str(i)+"_"+str(self.parameters['DT__class_weight'][0])] = (fit, sel, sel_var)


    def __foresee_basics(self, X, y, A, unpriv_class):
        '''
        Computes the  leaves, level of discrimination in each leave, and the index of the leave for each
        sample in the data train
        '''
        
        levels_disc = {} #level discriminations at leaves
        performances = [] #performances
        dt_indices = [] #dt indices
        
        for dt in self.k_dt:
            grid_model, sel, sel_var = self.k_dt[dt]
            X_sub = X[:,sel_var]

            #Get the predictions and leaves indices
            p, ld = self.__disc_leave(grid_model.best_estimator_['DT'], 
                                        X_sub, 
                                        A, 
                                        unpriv_class, 
                                        y)
            
            dt_indices.append(dt)
            performances.append(p)
            levels_disc[dt] = ld
        
        self.performances = dict(zip(np.array(dt_indices), performances))
        self.levels_discr = levels_disc
        
    def __disc_leave(self, model, X, A, unpriv_class, y):
        '''
        Returns the performances and level of discrimination in each leaf of the k-decision tree
        
        Inputs:
        models: set of decision trees trained on a data train. 
        X:  numpy (n,m), representing n samples of d dimension. The number of dimension should be consistent with models.
        y: numpy (n,), representing ground truth of dataset.
        A: numpy (n,), representing the membership of each group in A
        unpriv_class: string or value, representing the membership of the unprivileged group in A.
        
        Outputs:
        perfo: numpy (l_k,), representing the performance in each l_k leaf.
        level_disc: numpy (l_k, a), representing the level of discrimination for each group a in each l_k leaf.
        '''
        #Get the predictions and leave indices
        leaves = model.apply(X)
        y_pred = model.predict(X)

        #Compute the perfo of dt
        perfo = fm.f1score(y, y_pred)

        disc_leave = []
        for leaf in np.unique(leaves):
            misclassification_unpriv, misclassification_priv = fm.discrimination(np.array(y[leaves==leaf]).flatten(), 
                                                                                y_pred[leaves==leaf], 
                                                                                np.array(A[leaves==leaf]).flatten(), 
                                                                                unpriv_class= unpriv_class)

            disc_leave.append((misclassification_unpriv, misclassification_priv))

        disc_leave = np.array(disc_leave)
        level_disc = dict(zip(np.unique(leaves), disc_leave))

        return (perfo, level_disc)

    def predict(self, X, y, A, unpriv_class, alpha=0, beta=0):
        '''
        Return risk discrimination level for each sample in X
        
        Inputs:
        X:  numpy (n,m), representing n samples of d dimension. The number of dimension should be consistent with models.
        y: numpy (n,), representing ground truth of dataset.
        A: numpy (n,), representing the membership of each group in A
        priv_class: string or value, representing the membership of the privileged group in A.
        alpha: float, minimum level of discrimination for considering the leaf in risk computation
        beta: float, minimum level of performance for considering the leaf in risk computation
        
        Outputs:
        The following three dictionary contains the keys listed below, representing
        these scenarios:
            'total': all leaves.
            'th_alpha': only leaves with level of disc >=alpha
            'th_beta': only decision tress with perfo >= beta
            'th_both': only leaves with level of disc >=alpha & corresponding decision tree of perfo >= beta
        
        risks: dic, containing the risk discrimination to samples in X for each scenario.
        disc_acc: dic, containing the sum of discrimination to samples in X for each scenario.
        count_disc: dic, containing the leaf considered according alpha & beta to samples in X for each scenario.
        '''
        #Computation of thresholds
        threshold_disc=alpha
        threshold_perfo=beta


        #Computation of discrimination
        sum_total = np.zeros(y.shape[0])
        sum_thalpha = np.zeros(y.shape[0])
        sum_thbeta = np.zeros(y.shape[0])
        sum_thboth = np.zeros(y.shape[0])
        count_alpha = np.zeros(y.shape[0])
        count_beta = np.zeros(y.shape[0])
        count_both = np.zeros(y.shape[0])

        for index in self.k_dt:
            #Get leaves using the selected dimensions for k-it decision tree
            var_sel = self.k_dt[index][2] 
            leaves = self.k_dt[index][0].best_estimator_[0].apply(X[:,var_sel])
            
            #compute the risk of discrimination
            for leaf in self.levels_discr[index]:
                score = self.levels_discr[index][leaf]
                
                #risk of discrimination in in the leaf 
                risk_discr_m = abs(score[0]-score[1])
                
                #sum the discr level of the leaf to the accumulated discrimination level of each sample belonging to the leaf
                #the variable discr is of len equal to the number of samples in X
                discr = abs((1*(leaves==leaf))*(1*(A==unpriv_class))*risk_discr_m) + abs((1*(leaves==leaf))*(1*(A!=unpriv_class))*risk_discr_m)
                
                #Update the accumulated discr
                sum_total += discr
                
                #Update the accumulated discr above the alpha threshold
                if risk_discr_m>=threshold_disc:
                    sum_thalpha += discr
                    count_alpha += 1*(leaves==leaf)
                
                #Update the accumulated discr above beta threshold
                if self.performances[index]>=threshold_perfo:
                    sum_thbeta += discr
                    count_beta += 1*(leaves==leaf)
                
                #Update the accumulated discr above alpha and beta thresholds together
                if risk_discr_m>=threshold_disc:
                    if self.performances[index]>=threshold_perfo:
                        sum_thboth += discr
                        count_both += 1*(leaves==leaf)

        risk_total = sum_total/len(self.k_dt.keys())
        risk_thalpha = sum_thalpha/count_alpha
        risk_thbeta = sum_thbeta/count_beta
        risk_thboth = sum_thboth/count_both

        risks = {'total': risk_total,
                'th_alpha': risk_thalpha,
                'th_beta': risk_thbeta,
                'th_both': risk_thboth}

        disc_acc = {'total': sum_total,
                    'th_alpha': sum_thalpha,
                    'th_beta': sum_thbeta,
                    'th_both': sum_thboth}

        count_disc = {'total': len(self.k_dt.keys()),
                    'th_alpha': count_alpha,
                    'th_beta': count_beta,
                    'th_both': count_both}

        return (risks, disc_acc, count_disc)