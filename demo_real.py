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


def demonstration(dataset='h', clf='rf', n=3, costs=['TLPS'], ordering_cost='standard'):
    from utils import DatasetHelper

    D = DatasetHelper(dataset=dataset, feature_prefix_index=False)
    X_tr, X_ts, y_tr, y_ts = D.train_test_split()
    B, M = interaction_matrix(X_tr, interaction_type='causal')
    print('# Dataset: ', D.dataset_name)

    if(clf=='lr'):
        mdl = LogisticRegression(penalty='l2', C=1.0, fit_intercept=True, solver='liblinear', max_iter=10000)
        mdl = mdl.fit(X_tr, y_tr)
        oce = LinearOrderedActionExtractor(mdl, X_tr, feature_names=D.feature_names, feature_types=D.feature_types, feature_categories=D.feature_categories, 
                                           feature_constraints=D.feature_constraints, target_name=D.target_name, target_labels=D.target_labels, interaction_matrix=M)
        print('# Classifier: LogisticRegression')
    elif(clf=='mlp'):
        mdl = MLPClassifier(hidden_layer_sizes=(30,), max_iter=500, activation='relu', alpha=0.0001)
        mdl = mdl.fit(X_tr, y_tr)
        oce = MLPOrderedActionExtractor(mdl, X_tr, feature_names=D.feature_names, feature_types=D.feature_types, feature_categories=D.feature_categories,
                                        feature_constraints=D.feature_constraints, target_name=D.target_name, target_labels=D.target_labels, interaction_matrix=M, tol=tol)
        print('# Classifier: MultiLayerPerceptron')
    elif(clf=='rf'):
        h = 6 if dataset=='g' else 4
        mdl = RandomForestClassifier(n_estimators=30, max_depth=h)
        mdl = mdl.fit(X_tr, y_tr)
        oce = ForestOrderedActionExtractor(mdl, X_tr, feature_names=D.feature_names, feature_types=D.feature_types, feature_categories=D.feature_categories, 
                                           feature_constraints=D.feature_constraints, target_name=D.target_name, target_labels=D.target_labels, interaction_matrix=M)
        print('# Classifier: RandomForest')
    print()


    denied_individual = X_ts[mdl.predict(X_ts)==1]
    for i, x in enumerate(denied_individual[:n]):
        for cost in costs:
            # print('## {} (non-order):'.format(cost))
            # oa = oce.extract(x, K=5, ordering=False, post_ordering=False, cost_type=cost)
            # if(oa!=-1): print(oa)
            print('## {} + C_ord: '.format(cost))
            for g in [1.0]:
                oa = oce.extract(x, K=4, gamma=g, ordering=True, cost_type=cost, ordering_cost_type=ordering_cost, time_limit=60, log_stream=False)
                if(oa!=-1): print(oa)
        print()
    print('---')



if(__name__ == '__main__'):
    np.random.seed(0)

    demonstration(dataset='h', clf='rf', n=5, costs=['TLPS'], ordering_cost='standard')
