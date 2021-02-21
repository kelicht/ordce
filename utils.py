import numpy as np
import pandas as pd
from scipy.stats import median_abs_deviation as mad
from scipy.stats import gaussian_kde as kde
from scipy.interpolate import interp1d
from sklearn.covariance import EmpiricalCovariance, MinCovDet
from sklearn.model_selection import train_test_split
from lingam import DirectLiNGAM
from lingam.utils import make_dot
from itertools import permutations


def flatten(x): return sum(x, [])

def supp(a, tol=1e-8): return np.where(abs(a)>tol)[0]

def CumulativeDistributionFunction(x_d, X_d, l_buff=1e-6, r_buff=1e-6):
    kde_estimator = kde(X_d)
    pdf = kde_estimator(x_d)
    cdf_raw = np.cumsum(pdf)
    total = cdf_raw[-1] + l_buff + r_buff
    cdf = (l_buff + cdf_raw) / total
    percentile_ = interp1d(x=x_d, y=cdf, copy=False,fill_value=(l_buff,1.0-r_buff), bounds_error=False, assume_sorted=False)
    return percentile_

def cost_order(a, M, order, weights=[]):
    D = len(a)
    if(len(weights)!=D): weights = [1.0] * D
    ret = 0.0
    delta = [0.0] * D
    for d in order:
        eps = a[d] - delta[d]
        ret += weights[d] * abs(eps)
        for d_ in range(D): delta[d_] += M[d_][d] * eps
    return ret

def heuristic_ordering(a, M, weights=[], mode='greedy'):
    D = len(a)
    if(len(weights)!=D): weights = [1.0] * D
    support = list(supp(a))
    if(mode=='random'):
        order = np.random.permutation(support)
    elif(mode=='greedy'):
        order = []
        delta = [0.0] * D
        while(len(support)!=0):
            cands = []
            for d in support: cands.append(weights[d] * abs(a[d] - delta[d]))
            d_k = support.pop(np.argmin(cands))
            order.append(d_k)
            for d in range(D): delta[d] += M[d][d_k] * (a[d_k] - delta[d_k])
    return order

def cost_order_all_permutations(a, M, weights=[], stat_dict=False):
    D = len(a)
    support = list(supp(a))
    ret = []
    for order in permutations(support):
        c = cost_order(a, M, order, weights=weights)
        ret.append([order, c])
    costs = sorted(ret, key=lambda x:x[1])
    if(stat_dict):
        res_dict = {}
        res_dict['max'] = costs[-1][1]
        res_dict['min'] = costs[0][1]
        res_dict['mean'] = np.mean([c[1] for c in costs])
        res_dict['mean'] = np.std([c[1] for c in costs])
        return costs, res_dict
    else:
        return costs

def interaction_matrix(X, interaction_type='causal', prior_knowledge=None, measure='pwling', estimator='ML', file_name=''):
    if(interaction_type=='causal'):
        lingam = DirectLiNGAM(prior_knowledge=prior_knowledge, measure=measure).fit(X)
        B = lingam.adjacency_matrix_
        C = np.zeros([X.shape[1], X.shape[1]])
        for d in range(1, X.shape[1]): 
            C += np.linalg.matrix_power(B, d)
        return B, C
    elif(interaction_type=='correlation'):
        return np.corrcoef(X.T) - np.eye(X.shape[1])
    elif(interaction_type=='covariance'):
        if(estimator=='ML'):
            est = EmpiricalCovariance(store_precision=True, assume_centered=False).fit(X)
        elif(estimator=='MCD'):
            est = MinCovDet(store_precision=True, assume_centered=False, support_fraction=None).fit(X)
        cov = est.covariance_
        if(np.linalg.matrix_rank(cov)!=X.shape[1]): cov += 1e-6 * np.eye(X.shape[1])
        l_, P_ = np.linalg.eig(np.linalg.inv(cov))
        l = np.diag(np.sqrt(l_))
        P = P_.T
        U = P.T.dot(l).T
        return cov, U
    elif(interaction_type=='precomputed'):
        df = pd.read_csv(file_name)
        return df.values

class OrderedAction():
    def __init__(self, x, a, sig, gamma=-1, time=-1, obj=-1, c_ordinal=-1, c_ordering=-1,
                 target_name='Output', target_labels=['Good', 'Bad'], label_before=1, label_after=0, 
                 feature_names=[], feature_types=[], feature_categories=[], interaction_matrix=[], 
                 post_ordering=False, weights=[], post_ordering_mode='greedy'):
        self.x_ = x
        self.a_ = a
        self.sig_ = sig
        self.gamma_ = gamma
        self.time_ = time
        self.obj_ = obj
        self.c_ordinal_ = c_ordinal
        self.c_ordering_ = c_ordering
        self.target_name_ = target_name
        self.labels_ = [target_labels[label_before], target_labels[label_after]]
        self.feature_names_ = feature_names if len(feature_names)==len(x) else ['x_{}'.format(d) for d in range(len(x))]
        self.feature_types_ = feature_types if len(feature_types)==len(x) else ['C' for d in range(len(x))]
        self.feature_categories_ = feature_categories
        self.interaction_matrix_ = interaction_matrix

        self.feature_categories_inv_ = []
        for d in range(len(x)):
            g = -1
            if(self.feature_types_[d]=='B'):
                for i, cat in enumerate(self.feature_categories_):
                    if(d in cat): 
                        g = i
                        break
            self.feature_categories_inv_.append(g)            

        if(len(sig)!=0):
            self.Ks_, self.order_ = np.where(sig==1)
            self.c_ordering_ = cost_order(a, interaction_matrix, self.order_, weights=weights)
            self.ordered_ = True
        else:
            self.order_ = heuristic_ordering(a, interaction_matrix, weights=weights, mode=post_ordering_mode) if post_ordering else np.where(abs(a)>1e-8)[0]
            self.c_ordering_ = cost_order(a, interaction_matrix, self.order_, weights=weights)
            self.ordered_ = post_ordering

    def __str__(self):
        s = '* '
        if(self.ordered_==False): s += 'Non-'
        s += 'Ordered Action ({}: {} -> {})\n'.format(self.target_name_, self.labels_[0], self.labels_[1])
        i = 0
        for i,d in enumerate(self.order_):
            num = '* {}.'.format(i+1) if self.ordered_ else '*'
            g = self.feature_categories_inv_[d]
            if(g==-1):
                if(self.feature_types_[d]=='C'):
                    s += '\t{} {}: {:.4f} -> {:.4f} ({:+.4f})\n'.format(num, self.feature_names_[d], self.x_[d], self.x_[d]+self.a_[d], self.a_[d])
                else:
                    s += '\t{} {}: {} -> {} ({:+})\n'.format(num, self.feature_names_[d], self.x_[d].astype(int), (self.x_[d]+self.a_[d]).astype(int), self.a_[d].astype(int))
            else:
                if(self.x_[d]==1): continue
                cat_name, nxt = self.feature_names_[d].split(':')
                cat = self.feature_categories_[g]
                prv = self.feature_names_[cat[np.where(self.x_[cat])[0][0]]].split(':')[1]
                s += '\t{} {}: {} -> {}\n'.format(num, cat_name, prv, nxt)

        s += '\t* (Gamma = {0}, Obj. = {1:.4f} + {0} * {2:.4f} = {3:.4f}, Time = {4:.4f})'.format(self.gamma_, self.c_ordinal_, self.c_ordering_, self.obj_, self.time_)
        return s

    def getPartialOrder(self):
        total_order = self.order_
        K = len(total_order)
        flag = [0] * K
        partial_order = [[0] * K] * K
        return partial_order
    
    def a(self):
        return self.a_

    def order(self):
        return self.order_
# class OrderedAction


ACTION_TYPES = ['B', 'I', 'C']
ACTION_CONSTRAINTS = ['', 'FIX', 'INC', 'DEC']
class ActionCandidates():
    def __init__(self, X, feature_names=[], feature_types=[], feature_categories=[], feature_constraints=[], max_candidates=50, tol=1e-6):
        self.X_ = X
        self.N_, self.D_ = X.shape
        self.feature_names_ = feature_names if len(feature_names)==self.D_ else ['x_{}'.format(d) for d in range(self.D_)]
        self.feature_types_ = feature_types if len(feature_types)==self.D_ else ['C' for d in range(self.D_)]
        self.feature_categories_ = feature_categories
        self.feature_constraints_ = feature_constraints if len(feature_constraints)==self.D_ else ['' for d in range(self.D_)]
        self.max_candidates = max_candidates
        self.tol_ = tol

        self.X_lb_, self.X_ub_ = X.min(axis=0), X.max(axis=0)
        self.steps_ = [(self.X_ub_[d]-self.X_lb_[d])/max_candidates if self.feature_types_[d]=='C' else 1 for d in range(self.D_)]
        self.grids_ = [np.arange(self.X_lb_[d], self.X_ub_[d]+self.steps_[d], self.steps_[d]) for d in range(self.D_)]

        self.actions_ = None
        self.costs_ = None
        self.Q_ = None

    def getFeatureWeight(self, cost_type='uniform'):
        weights = np.ones(self.D_)
        if(cost_type=='MAD'):
            for d in range(self.D_):
                weight =  mad(self.X_[:,d], scale='normal')
                if(self.feature_types_[d]=='B' or abs(weight)<self.tol_):
                    weights[d] = (self.X_[:,d]*1.4826).std()
                else:
                    weights[d] = weight ** -1
        elif(cost_type=='standard'):
            weights = np.std(self.X_, axis=0) ** -1
        elif(cost_type=='normalize'):
            weights = (self.X_.max(axis=0) - self.X_.min(axis=0)) ** -1
        elif(cost_type=='robust'):
            q25, q75 = np.percentile(self.X_, [0.25, 0.75], axis=0)
            for d in range(self.D_):
                if(q75[d]-q25[d]==0):
                    weights[d] = self.tol_ ** -1
                else:
                    weights = (q75[d]-q25) ** -1
        return weights

    def setActionSet(self, x):
        self.actions_ = []
        for d in range(self.D_):
            if(self.feature_constraints_[d]=='FIX' or self.steps_[d] < self.tol_):
                self.actions_.append(np.array([]))
            elif(self.feature_types_[d]=='B'):
                if((self.feature_constraints_[d]=='INC' and x[d]==1) or (self.feature_constraints_[d]=='DEC' and x[d]==0)):
                    self.actions_.append(np.array([]))
                else:
                    self.actions_.append(np.array([ 1-2*x[d] ]))
            else:
                if(self.feature_constraints_[d]=='INC'):
                    start = x[d] + self.steps_[d]
                    stop = self.X_ub_[d] + self.steps_[d]
                elif(self.feature_constraints_[d]=='DEC'):
                    start = self.X_lb_[d]
                    stop = x[d]
                else:
                    start = self.X_lb_[d]
                    stop = self.X_ub_[d] + self.steps_[d]
                A_d = np.arange(start, stop, self.steps_[d]) - x[d]
                A_d = np.extract(abs(A_d)>self.tol_, A_d)
                if(len(A_d) > self.max_candidates): A_d = A_d[np.linspace(0, len(A_d)-1, self.max_candidates, dtype=int)]
                self.actions_.append(A_d)
        return self

    def setActionAndCost(self, x, cost_type='TLPS', p=1):
        self.costs_ = []
        self = self.setActionSet(x)
        if(cost_type=='TLPS'):
            if(self.Q_==None): self.Q_ = [None if self.feature_constraints_[d]=='FIX' else CumulativeDistributionFunction(self.grids_[d], self.X_[:, d]) for d in range(self.D_)]
            for d in range(self.D_):
                if(self.Q_[d]==None):
                    self.costs_.append([])
                else:
                    Q_d = self.Q_[d]
                    Q_0 = Q_d(x[d])
                    self.costs_.append( [ abs(np.log2( (1-Q_d(x[d]+a)) / (1-Q_0) )) for a in self.actions_[d] ] )
        elif(cost_type=='SCM' or cost_type=='DACE'):
            if(cost_type=='SCM'): 
                B_, _ = interaction_matrix(self.X_, interaction_type='causal') 
                # _, B_ = interaction_matrix(self.X_, interaction_type='causal')
                B = np.eye(self.D_) - B_
                C = self.getFeatureWeight(cost_type='standard')
            else:
                _, B = interaction_matrix(self.X_, interaction_type='covariance') 
                C = self.getFeatureWeight(cost_type='uniform')
            for d in range(self.D_):
                cost_d = []
                for d_ in range(self.D_): cost_d.append( [ C[d] * B[d][d_] * a  for a in self.actions_[d_] ] )
                self.costs_.append(cost_d)
        else:
            weights = self.getFeatureWeight(cost_type=cost_type)
            for d in range(self.D_):
                self.costs_.append( list(weights[d] * abs(self.actions_[d])**p) )
        return self

    def generateActions(self, x, cost_type='TLPS', p=1):
        self = self.setActionAndCost(x, cost_type=cost_type, p=p)
        return self.actions_, self.costs_
# class ActionCandidates


class ForestActionCandidates():
    def __init__(self, X, forest, feature_names=[], feature_types=[], feature_categories=[], feature_constraints=[], max_candidates=50, tol=1e-6):
        self.X_ = X
        self.N_, self.D_ = X.shape
        self.feature_names_ = feature_names if len(feature_names)==self.D_ else ['x_{}'.format(d) for d in range(self.D_)]
        self.feature_types_ = feature_types if len(feature_types)==self.D_ else ['C' for d in range(self.D_)]
        self.feature_categories_ = feature_categories
        self.feature_constraints_ = feature_constraints if len(feature_constraints)==self.D_ else ['' for d in range(self.D_)]
        self.tol_ = tol

        self.forest_ = forest
        self.T_ = forest.n_estimators
        self.trees_ = [t.tree_ for t in forest.estimators_]
        self.leaves_ = [np.where(tree.feature==-2)[0]  for tree in self.trees_]
        self.L_ = [len(l) for l in self.leaves_]

        self.H_ = self.getForestLabels()
        self.ancestors_, self.regions_ = self.getForestRegions()        
        self.thresholds_ = self.getForestThresholds()
        # self.M_ = [len(self.thresholds_[d])+1 for d in range(self.D_)]
        # self.partitions_ = self.getForestPartitions()

        self.X_lb_, self.X_ub_ = X.min(axis=0), X.max(axis=0)
        self.max_candidates = max_candidates
        self.steps_ = [(self.X_ub_[d]-self.X_lb_[d])/max_candidates if self.feature_types_[d]=='C' else 1 for d in range(self.D_)]
        self.grids_ = [np.arange(self.X_lb_[d], self.X_ub_[d]+self.steps_[d], self.steps_[d]) for d in range(self.D_)]

        self.x_ = None
        self.actions_ = None
        self.costs_ = None
        self.Q_ = None
        self.I_ = None

    def getForestLabels(self):
        H = []
        for tree, leaves, l_t in zip(self.trees_, self.leaves_, self.L_):
            falls = tree.value[leaves].reshape(l_t, 2)
            h_t = [val[0] if val.shape[0]==1 else val[1]/(val[0]+val[1]) for val in falls]
            H.append(h_t)
        return H

    def getForestRegions(self):
        As, Rs = [], []
        for tree, leaves in zip(self.trees_, self.leaves_):
            A, R = [], []
            stack = [[]]
            L, U = [[-np.inf]*self.D_], [[np.inf]*self.D_]
            for n in range(tree.node_count):
                a, l, u = stack.pop(), L.pop(), U.pop()
                if(n in leaves):
                    A.append(a)
                    R.append([ (l[d], u[d]) for d in range(self.D_)])
                else:
                    d = tree.feature[n]
                    if(d not in a): a_ = list(a) + [d]
                    stack.append(a_); stack.append(a_)
                    # b = int(tree.threshold[n]) if self.feature_types_[d]=='I' else tree.threshold[n]
                    b = tree.threshold[n]
                    l_ = list(l); u_ = list(u)
                    l[d] = b; u[d] = b
                    U.append(u_); L.append(l)
                    U.append(u); L.append(l_)
            As.append(A); Rs.append(R)
        return As, Rs

    def getForestThresholds(self):
        B = []
        for d in range(self.D_):
            b_d = []
            for tree in self.trees_: 
                # b_d += list(tree.threshold[tree.feature==d].astype(int)) if self.feature_types_[d]=='I' else list(tree.threshold[tree.feature==d])
                b_d += list(tree.threshold[tree.feature==d])
            b_d = list(set(b_d))
            b_d.sort()
            B.append(np.array(b_d))
        return B

    def getForestPartitions(self):
        I = []
        for t in range(self.T_):
            I_t = []
            for l in range(self.L_[t]):
                I_t_l = []
                for d in range(self.D_):
                    if(self.regions_[t][l][d][0]==-np.inf):
                        start = 0
                    else:
                        start = self.thresholds_[d].index(self.regions_[t][l][d][0]) + 1
                    if(self.regions_[t][l][d][1]== np.inf):
                        end = self.M_[d]
                    else:
                        end = self.thresholds_[d].index(self.regions_[t][l][d][1]) + 1
                    tmp = list(range(start, end))
                    I_t_l.append(tmp)
                I_t.append(I_t_l)
            I.append(I_t)
        return I

    def getFeatureWeight(self, cost_type='uniform'):
        weights = np.ones(self.D_)
        if(cost_type=='MAD'):
            from scipy.stats import median_abs_deviation as mad
            for d in range(self.D_):
                weight =  mad(self.X_[:,d], scale='normal')
                if(self.feature_types_[d]=='B' or abs(weight)<self.tol_):
                    weights[d] = (self.X_[:,d]*1.4826).std()
                else:
                    weights[d] = weight ** -1
        elif(cost_type=='standard'):
            weights = np.std(self.X_, axis=0) ** -1
        elif(cost_type=='normalize'):
            weights = (self.X_.max(axis=0) - self.X_.min(axis=0)) ** -1
        elif(cost_type=='robust'):
            weights = (np.quantile(self.X_, 0.75, axis=0) - np.quantile(self.X_, 0.25, axis=0)) ** -1
        return weights

    def setActionSet(self, x, use_threshold=True):
        if((x == self.x_).all()): return self
        self.x_ = x
        self.actions_ = []
        for d in range(self.D_):
            if(self.feature_constraints_[d]=='FIX' or self.steps_[d] < self.tol_):
                self.actions_.append(np.array([ 0 ]))
            elif(self.feature_types_[d]=='B'):
                if((self.feature_constraints_[d]=='INC' and x[d]==1) or (self.feature_constraints_[d]=='DEC' and x[d]==0)):
                    self.actions_.append(np.array([ 0 ]))
                else:
                    self.actions_.append(np.array([ 1-2*x[d], 0 ]))
            else:
                if(use_threshold):
                    A_d = self.thresholds_[d].astype(int) - x[d] if self.feature_types_[d]=='I' else self.thresholds_[d] - x[d]
                    A_d[A_d>=0] += self.tol_ if self.feature_types_[d]=='C' else 1
                    if(0 not in A_d): A_d = np.append(A_d, 0)
                    if(self.feature_constraints_[d]=='INC'): 
                        A_d = np.extract(A_d>=0, A_d)
                    elif(self.feature_constraints_[d]=='DEC'): 
                        A_d = np.extract(A_d<=0, A_d)
                else:
                    if(self.feature_constraints_[d]=='INC'):
                        start = x[d] + self.steps_[d]
                        stop = self.X_ub_[d] + self.steps_[d]
                    elif(self.feature_constraints_[d]=='DEC'):
                        start = self.X_lb_[d]
                        stop = x[d]
                    else:
                        start = self.X_lb_[d]
                        stop = self.X_ub_[d] + self.steps_[d]
                    A_d = np.arange(start, stop, self.steps_[d]) - x[d]
                    A_d = np.extract(abs(A_d)>self.tol_, A_d)
                    if(len(A_d) > self.max_candidates): A_d = A_d[np.linspace(0, len(A_d)-1, self.max_candidates, dtype=int)]
                    A_d = np.append(A_d, 0)
                self.actions_.append(A_d)
        self = self.setForestIntervals(x)
        return self

    def setActionAndCost(self, x, cost_type='TLPS', p=1, use_threshold=True):
        self.costs_ = []
        self = self.setActionSet(x, use_threshold=use_threshold)
        if(cost_type=='TLPS'):
            if(self.Q_==None): self.Q_ = [None if self.feature_constraints_[d]=='FIX' else CumulativeDistributionFunction(self.grids_[d], self.X_[:, d]) for d in range(self.D_)]
            for d in range(self.D_):
                if(self.Q_[d]==None):
                    self.costs_.append([ 0 ])
                else:
                    Q_d = self.Q_[d]
                    Q_0 = Q_d(x[d])
                    self.costs_.append( [ abs(np.log2( (1-Q_d(x[d]+a)) / (1-Q_0) )) for a in self.actions_[d] ] )
        elif(cost_type=='SCM' or cost_type=='DACE'):
            if(cost_type=='SCM'): 
                B_, _ = interaction_matrix(self.X_, interaction_type='causal') 
                # _, B_ = interaction_matrix(self.X_, interaction_type='causal')
                B = np.eye(self.D_) - B_
                C = self.getFeatureWeight(cost_type='standard')
            else:
                _, B = interaction_matrix(self.X_, interaction_type='covariance') 
                C = self.getFeatureWeight(cost_type='uniform')
            for d in range(self.D_):
                cost_d = []
                for d_ in range(self.D_): cost_d.append( [ C[d] * B[d][d_] * a  for a in self.actions_[d_] ] )
                self.costs_.append(cost_d)
        else:
            weights = self.getFeatureWeight(cost_type=cost_type)
            for d in range(self.D_):
                self.costs_.append( list(weights[d] * abs(self.actions_[d])**p) )
        return self

    def setForestIntervals(self, x):
        Is = [np.arange(len(a)) for a in self.actions_]
        I = []
        for t in range(self.T_):
            I_t = []
            for l in range(self.L_[t]):
                I_t_l = []
                for d in range(self.D_):
                    xa = x[d] + self.actions_[d]
                    I_t_l.append( list(((xa > self.regions_[t][l][d][0]) & (xa <= self.regions_[t][l][d][1])).astype(int)) )
                    # I_t_l.append( (Is[d][ (xa > self.regions_[t][l][d][0]) & (xa <= self.regions_[t][l][d][1]) ]) )
                I_t.append(I_t_l)
            I.append(I_t)
        self.I_ = I
        return self

    def generateActions(self, x, cost_type='TLPS', p=1, use_threshold=True):
        self = self.setActionAndCost(x, cost_type=cost_type, p=p, use_threshold=use_threshold)
        return self.actions_, self.costs_, self.I_

# class ForestActionCandidates


DATASETS = ['g', 'w', 'h', 'c', 'a', 'd']
DATASETS_NAME = {
    'g':'german',
    'w':'wine',
    'h':'fico',
    'c':'compas',
    'a':'adult', 
    'd':'diabetes',
    's':'synthetic',
}
DATASETS_PATH = {
    # 'g':'data/german_credit.csv',
    'g':'data/german_processed.csv',
    'w':'data/wine.csv',
    'h':'data/heloc.csv',
    'c':'data/compas.csv',
    'a':'data/adult_onehot.csv', 
    'd':'data/diabetes.csv',
    's':'data/synthetic.csv',
}
TARGET_NAME = {
    # 'g':'Default',
    'g':'GoodCustomer',
    'w':'Quality',
    'h':'RiskPerformance',
    'c':'RecidivateWithinTwoYears',
    'a':'Income', 
    'd':'Outcome',
    's':'Loan',
}
TARGET_LABELS = {
    'g':['Good', 'Bad'],
    'w':['>5', '<=5'],
    'h':['Good', 'Bad'],
    'c':['NotRecidivate','Recidivate'],
    'a':['>50K','<=50K'], 
    'd':['Good', 'Bad'],
    's':['Accept', 'Reject'],
}
FEATURE_NUMS = {
    'g':40,
    'w':12,
    'h':23,
    'c':14,
    'a':108,
    'd':8,
    's':5,
}
FEATURE_TYPES = {
    'g':['I']*7 + ['B']*33,
    'w':['C']*5 + ['I']*2 + ['C']*4 + ['B'],
    'h':['I']*23,
    'c':['I']*5 + ['B']*9,
    'a':['I']*6 + ['B']*102,
    'd':['I'] + ['C']*6 + ['I'],
    's':['I']*5,
}
FEATURE_CATEGORIES = {
    'g':[[18,19,20,21,22,23,24,25,26,27],[28,29,30],[31,32,33],[34,35,36]],
    'w':[],
    'h':[],
    'c':[[5,6,7,8,9,10],[11,12]],
    'a':[list(range(6,15)), list(range(15,31)), list(range(31,38)), list(range(38,53)),list(range(53,59))],
    'd':[],
    's':[],
}
FEATURE_CONSTRAINTS = {
    'g':['INC'] + ['']*36 + ['FIX']*3,
    'w':['']*11 + ['FIX'],
    'h':['']*23,
    'c':['INC'] + ['']*4 + ['FIX']*6 + ['']*2 + ['FIX'],
    'a':['INC'] + ['']*57 + ['FIX']*50,
    'd':['INC'] + ['']*6 + ['INC'],
    's':['INC']*2+['']*3,
}
class DatasetHelper():
    def __init__(self, dataset='h', feature_prefix_index=False):
        self.dataset_ = dataset
        self.df_ = pd.read_csv(DATASETS_PATH[dataset], dtype='float')
        self.y = self.df_[TARGET_NAME[dataset]].values
        self.X = self.df_.drop([TARGET_NAME[dataset]], axis=1).values
        self.feature_names = list(self.df_.drop([TARGET_NAME[dataset]], axis=1).columns)
        if(feature_prefix_index): self.feature_names = ['[x_{}]'.format(d) + feat for d,feat in enumerate(self.feature_names)]
        self.feature_types = FEATURE_TYPES[dataset]
        self.feature_categories = FEATURE_CATEGORIES[dataset]
        self.feature_constraints = FEATURE_CONSTRAINTS[dataset]
        self.target_name = TARGET_NAME[dataset]
        self.target_labels = TARGET_LABELS[dataset]
        self.dataset_name = DATASETS_NAME[dataset]

    def train_test_split(self, test_size=0.25):
        return train_test_split(self.X, self.y, test_size=test_size)

# class DatasetHelper


def _check_action_and_cost(dataset='h'):
    DH = DatasetHelper(dataset=dataset)
    AC = ActionCandidates(DH.X[1:], feature_names=DH.feature_names, feature_types=DH.feature_types, feature_categories=DH.feature_categories, feature_constraints=DH.feature_constraints)
    print(DH.X[0])
    A,C = AC.generateActions(DH.X[0], cost_type='TLPS')
    for d,a in zip(AC.feature_names_, A): print(d, a)
    for d,c in zip(AC.feature_names_, C): print(d, c)

def _check_forest_action(dataset='h'):
    from sklearn.ensemble import RandomForestClassifier
    DH = DatasetHelper(dataset=dataset)
    X,y = DH.X, DH.y
    forest = RandomForestClassifier(n_estimators=2, max_depth=4)
    forest = forest.fit(X[1:],y[1:])
    AC = ForestActionCandidates(DH.X[1:], forest, feature_names=DH.feature_names, feature_types=DH.feature_types, feature_categories=DH.feature_categories, feature_constraints=DH.feature_constraints)
    A, C, I, = AC.generateActions(X[0], cost_type='unifrom', use_threshold=True)
    # for d in range(X.shape[1]): print(d, A[d])
    # for d in range(X.shape[1]): print(d, C[d])
    for I_t in I:
        for I_t_l in I_t:
            print(I_t_l)
            print(flatten(I_t_l))

if(__name__ == '__main__'):
    # _check_action_and_cost(dataset='h')
    _check_forest_action(dataset='d')
