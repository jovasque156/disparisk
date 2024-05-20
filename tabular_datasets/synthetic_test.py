#Standard
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

import warnings
import pickle
warnings.filterwarnings('ignore')

# Own modules and libraries
import source.fairness as fm

import argparse

DIR_DATA = {
    'credit_card': 'data/credit_card/',
    'compas': 'data/compas/',
    'dutch_census': 'data/dutch_census/',
    'census_income':'data/census_income/',
    }

MODELS = {
    'credit_card': ['sgd_lr', 'mlp_one_layer', 'mlp_two_layer', 'mlp_three_layer'],
    'compas': ['sgd_lr', 'mlp_one_layer', 'mlp_two_layer', 'mlp_three_layer'],
    'dutch_census': ['sgd_lr', 'mlp_one_layer', 'mlp_two_layer', 'mlp_three_layer'],
    'census_income' : ['sgd_lr', 'mlp_one_layer', 'mlp_two_layer', 'mlp_three_layer'],
}

OVERREPRESENTED_CLASS = {
    'census_income': 0,
    'compas': 1, 
    'dutch_census': 0,
    'credit_card': 1 
}

def load_data():
    data_sets = {}
    for data in DIR_DATA:
        with open (DIR_DATA[data]+data+'.pkl', 'rb') as f:
            dic = pickle.load(f)
        
        data_sets[data] = dic
    return data_sets

#Sampling based on weights of classes
def sample_weights(X, S, Y, weights, stratifier=None):
    if stratifier is None:
        stratifier=Y
    
    n_samples = X.shape[0]
    n_classes = len(weights)
    n_samples_per_class = np.round(weights*n_samples).astype(int)
    n_samples_per_class[-1] = n_samples - np.sum(n_samples_per_class[:-1])
    X_s, S_s, Y_s = [], [], []
    for i in range(n_classes):
        idx = np.where(stratifier==i)[0]
        idx = np.random.choice(idx, n_samples_per_class[i], replace=True)
        X_s.append(X[idx])
        S_s.append(S[idx])
        Y_s.append(Y[idx])
    return np.concatenate(X_s), np.concatenate(S_s), np.concatenate(Y_s)

def compute_toussaint(V):
    left = np.log((2+V)/(2-V))-(2*V)/(2+V)
    right = (V**2)/2 + (V**4)/36 + (V**6)/288

    return max(left,right)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--it', type=str, default='1', help='iter_number')
    args = parser.parse_args()

    results = pd.DataFrame()
    
    print('Loading data sets')
    data_sets = load_data()

    for ds in DIR_DATA:
        print(f'Computing results for {ds}')
        count=1
        for weight in range(95, 4, -2):
            print(f'Computing results for weight {count} of {len(range(95,4,-2))}')
            X_train, S_train, Y_train = data_sets[ds]['train']
            X_test, S_test, Y_test = data_sets[ds]['test']
            features = data_sets[ds]['features'][0]+data_sets[ds]['features'][1]
            
            X_tr = np.concatenate((X_train.toarray(), X_test.toarray()), axis=0)
            S_tr = np.concatenate((S_train, S_test), axis=0)
            Y_tr = np.concatenate((Y_train, Y_test), axis=0)

            class_ = []
            for i in range(S_tr.shape[0]):
                    class_.append(S_tr[i]+Y_tr[i])

            X_s_1, S_s_1, Y_s_1 = X_tr[S_tr==OVERREPRESENTED_CLASS[ds]], S_tr[S_tr==OVERREPRESENTED_CLASS[ds]], Y_tr[S_tr==OVERREPRESENTED_CLASS[ds]]
            X_s_0, S_s_0, Y_s_0 = sample_weights(X_tr[S_tr!=OVERREPRESENTED_CLASS[ds]], S_tr[S_tr!=OVERREPRESENTED_CLASS[ds]], Y_tr[S_tr!=OVERREPRESENTED_CLASS[ds]], np.array([weight/100, 1.-weight/100]))

            X = np.concatenate((X_s_1, X_s_0), axis=0)
            S = np.concatenate((S_s_1, S_s_0), axis=0)
            Y = np.concatenate((Y_s_1, Y_s_0), axis=0)

            df = pd.concat([pd.DataFrame(X, columns=features), 
                            pd.DataFrame(S, columns=['sen']),
                            pd.DataFrame(Y, columns=['target'])],
                            axis=1, ignore_index=True)
            
            df.columns = features+['sen']+['target']

            df_train, df_test = train_test_split(df, test_size=.3, stratify=df['sen'])

            X_tr = df_train[features].to_numpy()
            S_tr = df_train['sen'].to_numpy()
            Y_tr = df_train['target'].to_numpy()

            X_te = df_train[features].to_numpy()
            S_te = df_train['sen'].to_numpy()
            Y_te = df_train['target'].to_numpy()

            for m in MODELS[ds]:
                print(f'Computing results for {m}   ', end='\r')
                
                S_mean = S_tr.mean()
                S_std = S_tr.std()
                Y_mean = Y_tr.mean()
                Y_std = Y_tr.std()
                
                #Load estimator
                estimator_s = pickle.load(open(f'v_info_scores/{m}/{ds}_s_v_info_estimator.pkl', 'rb'))
                model_s_v_info_estimator = estimator_s['estimator']
    
                #Compute prediction
                model_s_v_info_estimator.X_train = X_tr #np.concatenate((X_tr, ((S_tr-S_mean)/S_std).reshape(-1,1)), axis=1)
                model_s_v_info_estimator.model_v_C.fit(X_tr, Y_tr)
    
                Y_pred = model_s_v_info_estimator.model_v_C.predict(X_te)
                dem_p = fm.demographic_parity_dif(Y_pred, S_te, 1)
                dem_p_db = fm.demographic_parity_dif(Y_te, S_te, 1)

                # ipdb.set_trace()
                #Compute PVI(x->s)
                model_s_v_info_estimator.X_train = np.concatenate((X_tr, ((Y_tr-Y_mean)/Y_std).reshape(-1,1)), axis=1)
                model_s_v_info_estimator.Y_train = S_tr
                # model_s_v_info_estimator.model_v_C.fit(np.concatenate((X_tr, ((Y_tr-Y_mean)/Y_std).reshape(-1,1)), axis=1), S_tr)
    
                C = np.zeros(X_train.shape[1]+1)
                model_s_v_info_estimator.fit_on_C(C)
                pve_s = model_s_v_info_estimator.estimate_pve(S_te,
                                                        np.concatenate((X_te, ((Y_te-Y_mean)/Y_std).reshape(-1,1)), axis=1),
                                                        C)
                
                # ipdb.set_trace()
                C = np.ones(X_train.shape[1]+1)
                C[-1] = 0
                model_s_v_info_estimator.fit_on_C(C)
                pve_s_from_x = model_s_v_info_estimator.estimate_pve(S_te,
                                                                np.concatenate((X_te, ((Y_te-Y_mean)/Y_std).reshape(-1,1)), axis=1),
                                                                C)

                # ipdb.set_trace()
                pvi_s_from_x = pve_s - pve_s_from_x
                I_s_from_x = pvi_s_from_x.mean()
    
                g = (1-S_te.mean())*compute_toussaint(S_te.mean()*abs(dem_p))+S_te.mean()*compute_toussaint((1-S_te.mean())*abs(dem_p))
                g_ground_truth = (1-S_te.mean())*compute_toussaint(S_te.mean()*abs(dem_p_db))+S_te.mean()*compute_toussaint((1-S_te.mean())*abs(dem_p_db))
    
                results = pd.concat([results, pd.DataFrame({'dataset': ds, 
                                            'model': m,
                                            'pr(Y=1|S=0)': 1-weight/100,
                                            'DP': abs(dem_p),
                                            'DP_ground_truth': abs(dem_p_db),
                                            't(P(S=1), DP)': g,
                                            't(P(S=1), DP_ground_truth)': g_ground_truth,
                                            'I_v(X_to_S)': I_s_from_x}, index=[0])], axis=0, ignore_index=True)
                print('Saving results so far                                                        ')
                results.to_csv(f'lower_bounded_results_all_{args.it}.csv', index=False)

            count+=1

        print()
