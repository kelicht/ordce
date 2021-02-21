import numpy as np
from lingam import DirectLiNGAM
from lingam.utils import make_dot
from utils import interaction_matrix, cost_order_all_permutations
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from linear_oce import LinearOrderedActionExtractor
from mlp_oce import MLPOrderedActionExtractor
from forest_oce import ForestOrderedActionExtractor



def exp_synthetic(n=10, verbose=False):
    N = 1000
    c_21 = 1
    c_32 = 6
    c_34 = 4
    c_54 = -0.5
    names = ['Education','JobSkill','Income(K)','WorkPerDay','HealthStatus']
    x_1 = np.random.randint(1, 5, N)
    x_2 = c_21 * x_1 + np.random.randint(-1, 1, N)
    x_4 = np.random.randint(2, 6, N) * 2
    x_3 = c_32 * x_2 + c_34 * x_4 + np.random.randint(-2, 2, N)
    x_5 = c_54 * x_4 + np.random.randint(6, 13, N)
    X = np.array([x_1, x_2, x_3, x_4, x_5]).T
    _, C = interaction_matrix(X, interaction_type='causal')
    w_3, w_5 = 1.0/x_3.mean(), 1.0/x_5.mean()
    y = (w_3 * X[:,2] + w_5 * X[:,4] < 2.0).astype(int)

    mdl = LogisticRegression(penalty='l2', C=1.0, fit_intercept=True, solver='liblinear', max_iter=10000)
    mdl = mdl.fit(X, y)
    print('# Model Coef.: \n', mdl.coef_, (mdl.intercept_))
    oce = LinearOrderedActionExtractor(mdl, X, feature_names=names, feature_types=['I', 'I', 'I', 'I', 'I'], feature_constraints=['INC']*2+['']*3, target_name='Loan', target_labels=['Accept', 'Reject'], interaction_matrix=C)

    denied_individual = X[mdl.predict(X)==1]
    costs = ['TLPS', 'MAD', 'DACE', 'SCM']
    gammas = [1.0] if verbose else [0.1 + i * 0.1 for i in range(20)]
    res_dict = {}; res_dict_ord = {}; res_dict_time = {}
    for key in costs: 
        res_dict[key] = []; res_dict_ord[key] = []; res_dict_time[key] = []
    for c in costs:
        for g in gammas:
            key = c + '_ORDER_{}'.format(g)
            res_dict[key] = []; res_dict_ord[key] = []; res_dict_time[key] = []

    for i, x in enumerate(denied_individual[:n]):
        print('# {}-th Denied Individual: '.format(i+1), x)

        for cost in costs:
            print('## {}: '.format(cost))
            oa = oce.extract(x, K=5, ordering=False, post_ordering=True, post_ordering_mode='greedy', cost_type=cost, ordering_cost_type='standard')
            if(oa!=-1): 
                print(oa)
                res_dict[cost].append(oa.c_ordinal_)
                res_dict_ord[cost].append(oa.c_ordering_)
                res_dict_time[cost].append(oa.time_)

            if(verbose): print('## {} + C_order: '.format(cost))
            for gamma in gammas:
                oa = oce.extract(x, K=5, gamma=gamma, ordering=True, cost_type=cost, ordering_cost_type='standard')
                if(oa!=-1): 
                    print(oa)
                    res_dict[cost+'_ORDER_{}'.format(gamma)].append(oa.c_ordinal_)
                    res_dict_ord[cost+'_ORDER_{}'.format(gamma)].append(oa.c_ordering_)
                    res_dict_time[cost+'_ORDER_{}'.format(gamma)].append(oa.time_)
    
        print('---')

    if(verbose==False):
        import pandas as pd
        res_dist = pd.DataFrame(res_dict)
        res_dist.to_csv('./res/synthetic_res_dist_lr.csv', index=False)
        res_ord = pd.DataFrame(res_dict_ord)
        res_ord.to_csv('./res/synthetic_res_ord_lr.csv', index=False)
        res_time = pd.DataFrame(res_dict_time)
        res_time.to_csv('./res/synthetic_res_time_lr.csv', index=False)


def exp_real(clf='lr', dataset='h', n=10, verbose=False, costs=['TLPS','MAD','DACE','SCM'], suf='', tol=1e-6):
    from utils import DatasetHelper

    D = DatasetHelper(dataset=dataset, feature_prefix_index=False)
    X_tr, X_ts, y_tr, y_ts = D.train_test_split()
    B, M = interaction_matrix(X_tr, interaction_type='causal')

    if(clf=='lr'):
        mdl = LogisticRegression(penalty='l2', C=1.0, fit_intercept=True, solver='liblinear', max_iter=10000)
        mdl = mdl.fit(X_tr, y_tr)
        # print('# Model Coef.: \n', mdl.coef_, (mdl.intercept_))
        oce = LinearOrderedActionExtractor(mdl, X_tr, feature_names=D.feature_names, feature_types=D.feature_types, feature_categories=D.feature_categories, 
                                           feature_constraints=D.feature_constraints, target_name=D.target_name, target_labels=D.target_labels, interaction_matrix=M)
    elif(clf=='mlp'):
        mdl = MLPClassifier(hidden_layer_sizes=(200,), max_iter=500, activation='relu', alpha=0.0001)
        mdl = mdl.fit(X_tr, y_tr)
        oce = MLPOrderedActionExtractor(mdl, X_tr, feature_names=D.feature_names, feature_types=D.feature_types, feature_categories=D.feature_categories,
                                        feature_constraints=D.feature_constraints, target_name=D.target_name, target_labels=D.target_labels, interaction_matrix=M, tol=tol)
    elif(clf=='rf'):
        h = 6 if dataset=='g' else 4
        mdl = RandomForestClassifier(n_estimators=100, max_depth=h)
        mdl = mdl.fit(X_tr, y_tr)
        oce = ForestOrderedActionExtractor(mdl, X_tr, feature_names=D.feature_names, feature_types=D.feature_types, feature_categories=D.feature_categories, 
                                           feature_constraints=D.feature_constraints, target_name=D.target_name, target_labels=D.target_labels, interaction_matrix=M)

    denied_individual = X_ts[mdl.predict(X_ts)==1]
    gammas = [1.0]
    res_dict = {}; res_dict_ord = {}; res_dict_time = {}
    for key in costs: 
        res_dict[key] = []; res_dict_ord[key] = []; res_dict_time[key] = []
    for c in costs:
        for g in gammas:
            key = c + '_ORDER_{}'.format(g)
            res_dict[key] = []; res_dict_ord[key] = []; res_dict_time[key] = []

    for i, x in enumerate(denied_individual[:n]):
        print('# {}-th Denied Individual:'.format(i+1))

        for cost in costs:
            print('## {}: '.format(cost))
            oa = oce.extract(x, K=4, ordering=False, post_ordering=True, post_ordering_mode='greedy', cost_type=cost, ordering_cost_type='standard', time_limit=300, log_stream=False)
            if(oa!=-1): 
                print(oa)
                res_dict[cost].append(oa.c_ordinal_)
                res_dict_ord[cost].append(oa.c_ordering_)
                res_dict_time[cost].append(oa.time_)
            else:
                res_dict[cost].append(-1)
                res_dict_ord[cost].append(-1)
                res_dict_time[cost].append(-1)

            print('## {} + C_order: '.format(cost))
            for gamma in gammas:
                oa = oce.extract(x, K=4, gamma=gamma, ordering=True, cost_type=cost, ordering_cost_type='standard', time_limit=300, log_stream=False)
                if(oa!=-1): 
                    print(oa)
                    res_dict[cost+'_ORDER_{}'.format(gamma)].append(oa.c_ordinal_)
                    res_dict_ord[cost+'_ORDER_{}'.format(gamma)].append(oa.c_ordering_)
                    res_dict_time[cost+'_ORDER_{}'.format(gamma)].append(oa.time_)
                else:
                    res_dict[cost+'_ORDER_{}'.format(gamma)].append(-1)
                    res_dict_ord[cost+'_ORDER_{}'.format(gamma)].append(-1)
                    res_dict_time[cost+'_ORDER_{}'.format(gamma)].append(-1)
    
        if(verbose): print('---')

    print('# Results')
    print('+ ', res_dict)
    print('+ ', res_dict_ord)
    print('+ ', res_dict_time)
    print('---')

    if(verbose==False):
        import pandas as pd
        res_dist = pd.DataFrame(res_dict)
        res_dist.to_csv('./res/{}_res_dist_{}_{}.csv'.format(D.dataset_name, clf, suf), index=False)
        res_ord = pd.DataFrame(res_dict_ord)
        res_ord.to_csv('./res/{}_res_ord_{}_{}.csv'.format(D.dataset_name, clf, suf), index=False)
        res_time = pd.DataFrame(res_dict_time)
        res_time.to_csv('./res/{}_res_time_{}_{}.csv'.format(D.dataset_name, clf, suf), index=False)


def exp_real_sens(dataset='h', n=10, verbose=False, time_limit=300, costs=['TLPS', 'MAD', 'DACE', 'SCM']):
    from utils import DatasetHelper

    D = DatasetHelper(dataset=dataset, feature_prefix_index=False)
    X_tr, X_ts, y_tr, y_ts = D.train_test_split()
    B, M = interaction_matrix(X_tr, interaction_type='causal')

    mdl = LogisticRegression(penalty='l2', C=1.0, fit_intercept=True, solver='liblinear', max_iter=10000)
    mdl = mdl.fit(X_tr, y_tr)
    oce = LinearOrderedActionExtractor(mdl, X_tr, feature_names=D.feature_names, feature_types=D.feature_types, feature_categories=D.feature_categories, 
                                        feature_constraints=D.feature_constraints, target_name=D.target_name, target_labels=D.target_labels, interaction_matrix=M)

    denied_individual = X_ts[mdl.predict(X_ts)==1]
    gammas = [10**i for i in range(-3, 3)]
    res_dict = {}; res_dict_ord = {}
    for key in costs: 
        res_dict[key] = []; res_dict_ord[key] = []
    for c in costs:
        for g in gammas:
            key = c + '_ORDER_{}'.format(g)
            res_dict[key] = []; res_dict_ord[key] = []

    for i, x in enumerate(denied_individual[:n]):
        print('# {}-th Denied Individual:'.format(i+1))

        for cost in costs:
            print('## {}: '.format(cost))
            oa = oce.extract(x, K=4, ordering=False, post_ordering=True, post_ordering_mode='greedy', cost_type=cost, ordering_cost_type='standard', time_limit=time_limit)
            if(oa!=-1): 
                print(oa)
                res_dict[cost].append(oa.c_ordinal_)
                res_dict_ord[cost].append(oa.c_ordering_)
            else:
                res_dict[cost].append(-1)
                res_dict_ord[cost].append(-1)

            print('## {} + C_order: '.format(cost))
            for gamma in gammas:
                oa = oce.extract(x, K=4, gamma=gamma, ordering=True, cost_type=cost, ordering_cost_type='standard', time_limit=time_limit)
                if(oa!=-1): 
                    print(oa)
                    res_dict[cost+'_ORDER_{}'.format(gamma)].append(oa.c_ordinal_)
                    res_dict_ord[cost+'_ORDER_{}'.format(gamma)].append(oa.c_ordering_)
                else:
                    res_dict[cost+'_ORDER_{}'.format(gamma)].append(-1)
                    res_dict_ord[cost+'_ORDER_{}'.format(gamma)].append(-1)
    
        if(verbose): print('---')

    print('# Results')
    print('+ ', res_dict)
    print('+ ', res_dict_ord)
    print('---')

    if(verbose==False):
        import pandas as pd
        res_dist = pd.DataFrame(res_dict)
        res_dist.to_csv('./res/{}_res_dist_sens.csv'.format(D.dataset_name), index=False)
        res_ord = pd.DataFrame(res_dict_ord)
        res_ord.to_csv('./res/{}_res_ord_sens.csv'.format(D.dataset_name), index=False)





if(__name__ == '__main__'):
    np.random.seed(1)

    for dataset in ['g', 'd', 'w', 'h']:
        exp_real(clf='lr', dataset=dataset, n=50, costs=['TLPS', 'DACE'])

    # for dataset in ['g', 'd', 'w', 'h']:
    #     exp_real(clf='mlp', dataset=dataset, n=50, costs=['TLPS', 'DACE'])

    # for dataset in ['g', 'd', 'w', 'h']:
    #     exp_real(clf='rf', dataset=dataset, n=50, costs=['TLPS', 'DACE'])

    # for dataset in ['g', 'd', 'w', 'h']:
    #     exp_real_sens(dataset=dataset, n=50, time_limit=60, costs=['TLPS', 'DACE'])


