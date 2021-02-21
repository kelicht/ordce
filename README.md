# OrdCE: Ordered Counterfactual Explanation

Temporary repository for our paper: "*Ordered Counterfactual Explanation by Mixed-Integer Linear Optimization*" (to appear in AAAI-21, our preprint is available [here](https://arxiv.org/abs/2012.11782))

OrdCE provides an *ordered action* (a, s), which is a pair of an *action (perturbation vector)* a and its *perutrbing order* s. 

OrdCE optimizes the following objective function: C(a, s | x) = C_dist(a) + gamma * C_ord(a, s), where
* C_dist(a): distance-based cost function (i.e., objective function of existing CE methods such as *total-log percentile shift*[Ustun+, FAT*'19])
* C_ord(a, s): ordering cost function to determine an optimal perturbing order for a given action (our proposed)
* gamma: Trade-off parameter between C_dist and C_ord

Running examples are provided in `demo_synthetic.py` and `demo_real.py`.
```
$ python demo_synthetic.py 
# Synthetic Example:
	- x_1: Education       : 1 ~ 5
	- x_2: JobSkill        : 1 ~ 10
	- x_3: Income          : 10 ~ 100
	- x_4: WorkPerDay       : 4 ~ 10
	- x_5: HealthStatus : 1 ~ 10
# Causal Relationship:
	- x_1 = e_1
	- x_2 = 1 * x_1 + e_2
	- x_3 = 6 * x_2 + 4 * x_4 + e_3
	- x_4 = e_4
	- x_5 = -0.5 * x_4 + e_5
# Interaction Matrix: 
 [[ 0.   0.   0.   0.   0. ]
 [ 1.   0.   0.   0.   0. ]
 [ 6.   6.   0.   4.   0. ]
 [ 0.   0.   0.   0.   0. ]
 [ 0.   0.   0.  -0.5  0. ]]
# Classifier: LogisticRegression

# 1-th Denied Individual: 
	- x_1: Education       : 1.0
	- x_2: JobSkill        : 1.0
	- x_3: Income          : 29.0
	- x_4: WorkPerDay      : 6.0
	- x_5: HealthStatus    : 6.0
## TLPS + C_ord: 
* Ordered Action (Loan: Reject -> Accept)
	* 1. WorkPerDay: 6 -> 7 (+1)
	* 2. Income: 29 -> 33 (+4)
	* (Gamma = 1.0, Obj. = 0.4079 + 1.0 * 1.0000 = 1.4079, Time = 0.0636)

---
```

```
$ python demo_real.py 
# Dataset:  fico
# Classifier: RandomForest

## TLPS + C_ord: 
* Ordered Action (RiskPerformance: Bad -> Good)
	* 1. NetFractionRevolvingBurden: 40 -> 35 (-5)
	* 2. AverageMInFile: 42 -> 60 (+18)
	* 3. MaxDelq2PublicRecLast12M: 4 -> 6 (+2)
	* (Gamma = 1.0, Obj. = 1.3153 + 1.0 * 1.8452 = 3.1605, Time = 23.5568)

## TLPS + C_ord: 
* Ordered Action (RiskPerformance: Bad -> Good)
	* 1. MaxDelq2PublicRecLast12M: 4 -> 5 (+1)
	* 2. MaxDelqEver: 6 -> 7 (+1)
	* 3. MSinceOldestTradeOpen: 118 -> 119 (+1)
	* 4. ExternalRiskEstimate: 62 -> 74 (+12)
	* (Gamma = 1.0, Obj. = 1.0888 + 1.0 * 1.6558 = 2.7446, Time = 15.1959)

## TLPS + C_ord: 
* Ordered Action (RiskPerformance: Bad -> Good)
	* 1. AverageMInFile: 57 -> 60 (+3)
	* 2. ExternalRiskEstimate: 63 -> 78 (+15)
	* (Gamma = 1.0, Obj. = 1.4796 + 1.0 * 1.5395 = 3.0191, Time = 5.5600)

## TLPS + C_ord: 
* Ordered Action (RiskPerformance: Bad -> Good)
	* 1. NetFractionRevolvingBurden: 43 -> 37 (-6)
	* 2. NumTotalTrades: 21 -> 22 (+1)
	* 3. AverageMInFile: 50 -> 51 (+1)
	* (Gamma = 1.0, Obj. = 0.3333 + 1.0 * 0.2538 = 0.5871, Time = 2.1308)

## TLPS + C_ord: 
* Ordered Action (RiskPerformance: Bad -> Good)
	* 1. ExternalRiskEstimate: 73 -> 74 (+1)
	* (Gamma = 1.0, Obj. = 0.1094 + 1.0 * 0.0981 = 0.2075, Time = 0.5840)

---
```
