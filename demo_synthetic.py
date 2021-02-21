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


def demonstration(clf='lr', n=1, costs=['TLPS'], ordering_cost='uniform', verbose=True):
    N = 1000
    c_21 = 1
    c_32 = 6
    c_34 = 4
    c_54 = -0.5
    names = ['Education','JobSkill','Income','WorkPerDay','HealthStatus']
    x_1 = np.random.randint(1, 5, N)
    x_2 = c_21 * x_1 + np.random.randint(-1, 1, N)
    x_4 = np.random.randint(2, 6, N) * 2
    x_3 = c_32 * x_2 + c_34 * x_4 + np.random.randint(-2, 2, N)
    x_5 = c_54 * x_4 + np.random.randint(6, 13, N)
    X = np.array([x_1, x_2, x_3, x_4, x_5]).T
    _, C = interaction_matrix(X, interaction_type='causal')
    w_3, w_5 = 1.0/x_3.mean(), 1.0/x_5.mean()
    y = (w_3 * X[:,2] + w_5 * X[:,4] < 2.0).astype(int)

    if(verbose):
        print('# Synthetic Example:')
        print('\t- x_1: Education       : 1 ~ 5')
        print('\t- x_2: JobSkill        : 1 ~ 10')
        print('\t- x_3: Income          : 10 ~ 100')
        print('\t- x_4: WorkPerDay       : 4 ~ 10')
        print('\t- x_5: HealthStatus : 1 ~ 10')
        print('# Causal Relationship:')
        print('\t- x_1 = e_1')
        print('\t- x_2 = {} * x_1 + e_2'.format(c_21))
        print('\t- x_3 = {} * x_2 + {} * x_4 + e_3'.format(c_32, c_34))
        print('\t- x_4 = e_4')
        print('\t- x_5 = {} * x_4 + e_5'.format(c_54))
        C = np.array([[0.0]*5, [c_21]+[0.0]*4, [c_21*c_32,c_32,0.0,c_34,0.0], [0.0]*5, [0.0]*3+[c_54,0.0]])
        print('# Interaction Matrix: \n', C)
        B = np.array([[0.0]*5, [c_21]+[0.0]*4, [0.0,c_32,0.0,c_34,0.0], [0.0]*5, [0.0]*3+[c_54,0.0]])
        # dot = make_dot(B, labels=names)
        # dot.attr('graph', fontname='arial')
        # dot.render('demonstration_causal_dag')

    if(clf=='lr'):
        mdl = LogisticRegression(penalty='l2', C=1.0, fit_intercept=True, solver='liblinear', max_iter=10000)
        mdl = mdl.fit(X, y)
        # print('# Model Coef.: \n', mdl.coef_, (mdl.intercept_))
        oce = LinearOrderedActionExtractor(mdl, X, feature_names=names, feature_types=['I', 'I', 'I', 'I', 'I'], feature_constraints=['INC']*2+['']*3, target_name='Loan', target_labels=['Accept', 'Reject'], interaction_matrix=C)
        print('# Classifier: LogisticRegression')
    elif(clf=='mlp'):
        mdl = MLPClassifier(hidden_layer_sizes=(30,), max_iter=500, activation='relu', alpha=0.0001)
        mdl = mdl.fit(X, y)
        oce = MLPOrderedActionExtractor(mdl, X, feature_names=names, feature_types=['I', 'I', 'I', 'I', 'I'], feature_constraints=['INC']*2+['']*3, target_name='Loan', target_labels=['Accept', 'Reject'], interaction_matrix=C)
        print('# Classifier: MultiLayerPerceptron')
    elif(clf=='rf'):
        mdl = RandomForestClassifier(n_estimators=30, max_depth=6)
        mdl = mdl.fit(X, y)
        oce = ForestOrderedActionExtractor(mdl, X, feature_names=names, feature_types=['I', 'I', 'I', 'I', 'I'], feature_constraints=['INC']*2+['']*3, target_name='Loan', target_labels=['Accept', 'Reject'], interaction_matrix=C)
        print('# Classifier: RandomForest')
    print()

    denied_individual = X[mdl.predict(X)==1]
    for i, x in enumerate(denied_individual[:n]):
        print('# {}-th Denied Individual: '.format(i+1))
        print('\t- x_1: Education       : {}'.format(x[0]))
        print('\t- x_2: JobSkill        : {}'.format(x[1]))
        print('\t- x_3: Income          : {}'.format(x[2]))
        print('\t- x_4: WorkPerDay      : {}'.format(x[3]))
        print('\t- x_5: HealthStatus    : {}'.format(x[4]))
        for cost in costs:
            print('## {} (non-order):'.format(cost))
            oa = oce.extract(x, K=5, ordering=False, post_ordering=False, cost_type=cost)
            if(oa!=-1): print(oa)
            print('## {} + C_ord: '.format(cost))
            for g in [1.0]:
                oa = oce.extract(x, K=5, gamma=g, ordering=True, cost_type=cost, ordering_cost_type=ordering_cost)
                if(oa!=-1): print(oa)
        print()
    print('---')



if(__name__ == '__main__'):
    np.random.seed(0)

    demonstration(clf='lr', n=1, costs=['TLPS'])
