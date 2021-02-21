import numpy as np
import time
import pulp
from utils import flatten, interaction_matrix, OrderedAction, ForestActionCandidates
# np.set_printoptions(suppress=True, precision=3)


class ForestOrderedActionExtractor():
    def __init__(self, mdl, X, theta=0.5,
                 feature_names=[], feature_types=[], feature_categories=[], feature_constraints=[], max_cancidates=100, tol=1e-6,
                 target_name='Output', target_labels = ['Good','Bad'], interaction_matrix=[], 
                 ):
        self.mdl_ = mdl
        self.T_ = mdl.n_estimators
        self.coef_ = np.ones(self.T_) / self.T_
        self.intercept_ = -1 * theta
        self.X_ = X
        self.N_, self.D_ = X.shape

        self.feature_names_ = feature_names if len(feature_names)==self.D_ else ['x_{}'.format(d) for d in range(self.D_)]
        self.feature_types_ = feature_types if len(feature_types)==self.D_ else ['C' for d in range(self.D_)]
        self.feature_categories_ = feature_categories
        self.feature_categories_flatten_ = flatten(feature_categories)
        self.feature_constraints_ = feature_constraints if len(feature_constraints)==self.D_ else ['' for d in range(self.D_)]
        self.target_name_ = target_name
        self.target_labels_ = target_labels
        self.AC_ = ForestActionCandidates(X, mdl, feature_names=feature_names, feature_types=feature_types, feature_categories=feature_categories, feature_constraints=feature_constraints, max_candidates=max_cancidates, tol=tol)
        self.L_ = self.AC_.L_
        self.H_ = self.AC_.H_
        self.tol_ = tol
        self.M_ = interaction_matrix if len(interaction_matrix)==self.D_ else np.zeros([self.D_, self.D_])


    def getOrderingBounds(self, K, P):
        L, U = np.zeros([K, self.D_]), np.zeros([K, self.D_])
        for d in range(self.D_):
            L[0][d] = self.lb_[d]
            U[0][d] = self.ub_[d]
        for k in range(1, self.K_):
            for d in range(self.D_):
                L[k][d] = L[k-1][d] - np.max([np.max([P[d][d_]*L[k-1][d_], P[d][d_]*U[k-1][d_]]) for d_ in range(self.D_)])
                U[k][d] = U[k-1][d] - np.min([np.min([P[d][d_]*L[k-1][d_], P[d][d_]*U[k-1][d_]]) for d_ in range(self.D_)])
        return L-self.tol_, U+self.tol_

    def extract(self, x, 
                W=[], K=5, gamma=1.0, ordering=True, post_ordering=False, post_ordering_mode='greedy', intervention=False,
                cost_type='uniform', ordering_cost_type='uniform', use_threshold=True, 
                solver='cplex', time_limit=300, log_stream=False, mdl_name='', log_name='', init_sols={}, verbose=False):
        self.x_ = x
        self.y_ = self.mdl_.predict(x.reshape(1,-1))[0]

        self.W_ = W if len(W)!=0 else list(range(self.D_))
        self.K_ = min(K, self.D_)
        if(gamma==0.0): ordering = False
        if(ordering==False): gamma = 0.0
        self.gamma_ = gamma

        cost_type = 'normalize' if intervention else cost_type
        self.A_, self.C_, self.I_ = self.AC_.generateActions(x, cost_type=cost_type, use_threshold=use_threshold)
        self.lb_ = [np.min(A_d) for A_d in self.A_]
        self.ub_ = [np.max(A_d) for A_d in self.A_]
        C = self.M_
        prob = pulp.LpProblem(mdl_name)

        # variables
        act = [pulp.LpVariable('act_{}'.format(d), cat='Continuous', lowBound=self.lb_[d], upBound=self.ub_[d]) for d in range(self.D_)]
        pi = [[pulp.LpVariable('pi_{}_{}'.format(d,i), cat='Binary') for i in range(len(self.A_[d]))] for d in range(self.D_)]
        xi  = [pulp.LpVariable('xi_{}'.format(t), cat='Continuous', lowBound=0, upBound=1) for t in range(self.T_)]
        phi  = [[pulp.LpVariable('phi_{}_{}'.format(t,l), cat='Binary') for l in range(self.L_[t])] for t in range(self.T_)]
        if(ordering):
            LB, UB = self.getOrderingBounds(K, C)
            sigma = [[pulp.LpVariable('sig_{}_{}'.format(k,d), cat='Binary') for d in range(self.D_)] for k in range(self.K_)]
            pik = [[[pulp.LpVariable('pik_{}_{}_{}'.format(k,d,i), cat='Binary') for i in range(len(self.A_[d]))] for d in range(self.D_)] for k in range(self.K_)]
            epsilon = [[pulp.LpVariable('ips_{}_{}'.format(k,d), cat='Continuous') for d in range(self.D_)] for k in range(self.K_)]
            delta = [[pulp.LpVariable('dlt_{}_{}'.format(k,d), cat='Continuous') for d in range(self.D_)] for k in range(self.K_)]
            zeta = [pulp.LpVariable('zta_{}'.format(k), cat='Continuous', lowBound=0) for k in range(self.K_)]
        if(cost_type=='SCM' or cost_type=='DACE'): dist = [pulp.LpVariable('dist_{}'.format(d), cat='Continuous', lowBound=0) for d in range(self.D_)]

        # set initial values {val: [val_1, val_2, ...], ...}
        if(len(init_sols)!=0):
            for val, sols in init_sols.items():
                if(val=='act'):
                    for d,v in enumerate(sols): act[d].setInitialValue(v)
                elif(val=='pi'):
                    for d,vs in enumerate(sols):
                        for i,v in enumerate(vs):
                            pi[d][i].setInitialValue(v)
                elif(val=='xi'):
                    for d,v in enumerate(sols): xi[d].setInitialValue(v)
                elif(val=='phi'):
                    for t,vs in enumerate(sols): 
                        for l,v in enumerate(vs):
                            phi[t][l].setInitialValue(v)
                elif(val=='sigma'):
                    for l,d in enumerate(sols): 
                        for d_ in range(self.D_):
                            v = 1 if d_==d else 0
                            sigma[l][d].setInitialValue(v)
                    for k in range(l+1, self.K_):
                        for d_ in range(self.D_):
                            sigma[k][d].setInitialValue(0)

        # objective function
        if(ordering):
            if(cost_type=='NONE'):
                prob += pulp.lpDot([self.gamma_]*self.K_, zeta)
                prob += pulp.lpDot([self.gamma_]*self.K_, zeta) >= 0
            elif(cost_type=='SCM' or cost_type=='DACE'):
                prob += pulp.lpSum(dist) + pulp.lpDot([self.gamma_]*self.K_, zeta)
                prob += pulp.lpSum(dist) + pulp.lpDot([self.gamma_]*self.K_, zeta) >= 0
                for d in range(self.D_):
                    prob += dist[d] - pulp.lpDot(flatten(self.C_[d]), flatten(pi)) >= 0
                    prob += dist[d] + pulp.lpDot(flatten(self.C_[d]), flatten(pi)) >= 0
            else:
                prob += pulp.lpDot(flatten(self.C_), flatten(pi)) + pulp.lpDot([self.gamma_]*self.K_, zeta)
                prob += pulp.lpDot(flatten(self.C_), flatten(pi)) + pulp.lpDot([self.gamma_]*self.K_, zeta) >= 0
        else:
            if(cost_type=='SCM' or cost_type=='DACE'):
                prob += pulp.lpSum(dist)
                prob += pulp.lpSum(dist) >= 0
                for d in range(self.D_):
                    prob += dist[d] - pulp.lpDot(flatten(self.C_[d]), flatten(pi)) >= 0
                    prob += dist[d] + pulp.lpDot(flatten(self.C_[d]), flatten(pi)) >= 0
            else:
                prob += pulp.lpDot(flatten(self.C_), flatten(pi))
                prob += pulp.lpDot(flatten(self.C_), flatten(pi)) >= 0

        # constraint: sum_{i} pi_{d,i} == 1
        for d in range(self.D_): prob += pulp.lpSum(pi[d]) == 1

        # constraint: sum_{d} pi_{d,i_d} >= D - K
        prob += pulp.lpSum([pi[d][list(self.A_[d]).index(0)] for d in range(self.D_)]) >= self.D_ - self.K_

        # constraint: sum_{d in G} a_{d,1} pi_{d,1} = 0
        for G in self.feature_categories_: prob += pulp.lpDot([self.A_[d][0] for d in G if len(self.A_[d])!=0], [pi[d][0] for d in G if len(self.A_[d])!=0]) == 0

        # constraint: sum_{d} w_d xi_d + b >= 0
        if(self.y_ == 0):
            prob += pulp.lpDot(self.coef_, xi) >= - self.intercept_ + 1e-8
        else:
            prob += pulp.lpDot(self.coef_, xi) <= - self.intercept_ - 1e-8

        # constraint: a_d = sum_{i} a_{d,i} pi_{d,i}
        if(intervention):
            _, B_ = interaction_matrix(self.X_, interaction_type='causal')
            # B_, _ = interaction_matrix(self.X_, interaction_type='causal')
            B = B_ + np.eye(self.D_)
            for d in range(self.D_):
                A_d = [ [ B[d][d_] * a for a in self.A_[d_] ] for d_ in range(self.D_) ]
                prob += act[d] - pulp.lpDot(flatten(A_d), flatten(pi)) == 0
        else:
            for d in range(self.D_): prob += act[d] - pulp.lpDot(self.A_[d], pi[d]) == 0

        # constraints (Tree Ensemble):
        for t in range(self.T_):
            # constraint: sum_{l} phi_{t,l} = 1
            prob += pulp.lpSum(phi[t]) == 1
            # constraint: xi_t = sum_{l} h_{t,l} phi_{t,l}
            prob += xi[t] - pulp.lpDot(self.H_[t], phi[t]) == 0
            # constraint: D * phi_{t,l} <= sum_{d} sum_{i in I_{t,l,d}} pi_{d,i}
            for l in range(self.L_[t]):
                anc = self.AC_.ancestors_[t][l]
                prob += len(anc) * phi[t][l] - pulp.lpDot(flatten([self.I_[t][l][d] for d in anc]), flatten([pi[d] for d in anc])) <= 0

        if(ordering):
            I_act = [(abs(a)>self.tol_).astype(int) for a in self.A_]

            # constraint: sigma_{k,d} = sum_{i} pi^(k)_{d,i}
            for k in range(self.K_):
                for d in range(self.D_):
                    if(d in self.feature_categories_flatten_ and self.x_[d]==1):
                        prob += sigma[k][d] == 0
                    else:
                        prob += pulp.lpDot(I_act[d], pik[k][d]) - sigma[k][d]  == 0

            # constraint: pis_{d,i} = sum_{k} pi^(k)_{d,i}
            for d in range(self.D_):
                for i in range(len(self.A_[d])):
                        prob += pulp.lpSum([pik[k][d][i] for k in range(self.K_)]) - pi[d][i]  == 0

            # constraint: sum_{d in G} a_{d,1} pi^(k)_{d,1} = 0
            for G in self.feature_categories_: prob += pulp.lpDot([self.A_[d][0] for d in G if len(self.A_[d])!=0], [pik[k][d][0] for d in G if len(self.A_[d])!=0]) == 0

            # constraint: sum_{k} sigma_{k,d}  <= 1
            for d in range(self.D_): prob += pulp.lpSum([sigma[k][d] for k in range(self.K_)]) <= 1

            # constraint: sum_{d} sigma_{k,d}  <= 1
            for k in range(self.K_): prob += pulp.lpSum(sigma[k]) <= 1

            # constraint: sum_{d} sigma_{k,d}  >= sum_{d} sigma_{k+1,d}
            for k in range(self.K_-1): prob += pulp.lpSum(sigma[k]) - pulp.lpSum(sigma[k+1]) >= 0

            # constraint: delta_{0,d} = 0
            for d in range(self.D_): prob += delta[0][d] == 0

            # constraint: delta_{k,d} = sum_{l}^{k-1} sum_{d'} C_{d,d'} epsilon_{l,d'} = sum_{d'} C_{d,d'} epsilon_{k-1,d'} + delta_{k-1,d}
            for k in range(1,self.K_):
                for d in range(self.D_):
                    # prob += delta[k][d] - pulp.lpDot(list(C[d])*(k), flatten(epsilon[:k+1])) == 0
                    prob += delta[k][d] - delta[k-1][d] - pulp.lpDot(C[d], epsilon[k-1]) == 0

            # constraint: epsilon_{k,d} >= sum_{i} a_{d,i} pi^(k)_{d,i} - delta_{k,d} - U_d (1 - sigma_{k,d})
            # constraint: epsilon_{k,d} <= sum_{i} a_{d,i} pi^(k)_{d,i} - delta_{k,d} - L_d (1 - sigma_{k,d})
            # constraint: epsilon_{k,d} >= L_d sigma_{k,d}
            # constraint: epsilon_{k,d} <= U_d sigma_{k,d}
            for k in range(self.K_):
                for d in range(self.D_):
                    prob += epsilon[k][d] - pulp.lpDot(self.A_[d], pik[k][d]) + delta[k][d] - UB[k][d] * sigma[k][d] >= - UB[k][d]
                    prob += epsilon[k][d] - pulp.lpDot(self.A_[d], pik[k][d]) + delta[k][d] - LB[k][d] * sigma[k][d] <= - LB[k][d]
                    prob += epsilon[k][d] - LB[k][d] * sigma[k][d] >= 0
                    prob += epsilon[k][d] - UB[k][d] * sigma[k][d] <= 0

            # constraint: zeta_k >= sum_{d} w_d * epsilon_{k,d}
            # constraint: zeta_k >= - sum_{d} w_d * epsilon_{k,d}
            weights = self.AC_.getFeatureWeight(cost_type=ordering_cost_type)
            for k in range(self.K_):
                prob += zeta[k] - pulp.lpDot(weights, epsilon[k]) >= 0
                prob += zeta[k] + pulp.lpDot(weights, epsilon[k]) >= 0


        if(len(log_name)!=0): prob.writeLP(log_name+'.lp')
        s = time.perf_counter()
        prob.solve(solver=pulp.CPLEX_PY(msg=log_stream, warm_start=(len(init_sols)!=0), timeLimit=time_limit, options=['set output clonelog -1']))
        t = time.perf_counter() - s
        if(prob.status!=1):
            # prob.solve(solver=pulp.CPLEX_PY(msg=True))
            return -1
        obj = prob.objective.value()
        if(cost_type=='SCM' or cost_type=='DACE'):
            c_ordinal = np.sum([d.value() for d in dist])
        else:
            c_ordinal = np.sum([c*round(p.value()) for c, p in zip(flatten(self.C_), flatten(pi))])
        c_ordering = np.sum([z.value() for z in zeta]) if ordering else 0.0

        a = np.array([ np.sum([ self.A_[d][i] * round(pi[d][i].value())  for i in range(len(self.A_[d])) ]) for d in range(self.D_) ])
        if(intervention): a_actual = np.array([a_.value() for a_ in act])
        sig = np.array([[round(s.value()) for s in sigma[k]] for k in range(self.K_)]) if ordering else []
        ret = OrderedAction(x, a, sig, gamma=gamma, time=t, obj=obj, c_ordinal=c_ordinal, c_ordering=c_ordering,
                            target_name=self.target_name_, target_labels=self.target_labels_, label_before=int(self.y_), label_after=int(1-self.y_), 
                            feature_names=self.feature_names_, feature_types=self.feature_types_, feature_categories=self.feature_categories_, interaction_matrix=self.M_,
                            post_ordering=post_ordering, weights=self.AC_.getFeatureWeight(cost_type=ordering_cost_type), post_ordering_mode=post_ordering_mode)

        # save initial values
        self.init_sols_ = {}
        self.init_sols_['act'] = [a_ for a_ in a]
        self.init_sols_['pi'] = []
        for pi_d in pi: self.init_sols_['pi'].append([round(p.value()) for p in pi_d])
        self.init_sols_['xi'] = [np.clip(x.value(), 0, 1) for x in xi]
        self.init_sols_['phi'] = []
        for phi_t in phi: self.init_sols_['phi'].append([round(p.value()) for p in phi_t])
        if(ret.ordered_): 
            self.init_sols_['sigma'] = ret.order_

        return ret



def _check_forest_oce(N=1, dataset='h', ordering = True):
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from utils import DatasetHelper, interaction_matrix
    np.random.seed(1)

    D = DatasetHelper(dataset=dataset, feature_prefix_index=False)
    X_tr, X_ts, y_tr, y_ts = D.train_test_split()
    mdl = RandomForestClassifier(n_estimators=100, max_depth=8)
    mdl = mdl.fit(X_tr, y_tr)
    denied = X_ts[mdl.predict(X_ts)==1]
    B, M = interaction_matrix(X_tr, interaction_type='causal')
    oce = ForestOrderedActionExtractor(mdl, X_tr, feature_names=D.feature_names, feature_types=D.feature_types, feature_categories=D.feature_categories, 
                                       feature_constraints=D.feature_constraints, target_name=D.target_name, target_labels=D.target_labels, interaction_matrix=M)

    for n,x in enumerate(denied[:N]):
        print('# {}-th Denied Individual ---------------------------'.format(n+1))
        oa = oce.extract(x, K=6, gamma=1.0, ordering=ordering, cost_type='uniform', ordering_cost_type='uniform', use_threshold=True, log_name='./lp/forest')
        if(oa!=-1): print(oa)


if(__name__ == '__main__'):
    _check_forest_oce(N=1, dataset='a', ordering=False)

