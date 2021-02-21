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
np.random.seed(1)


def plot_sens_select():
    import pandas as pd
    from matplotlib import pyplot as plt
    plt.rcParams["font.family"] = 'arial'
    plt.rcParams['text.usetex'] = True

    datas = ['fico','german','wine','diabetes']
    DATAS = {'fico':'FICO', 'german':'German', 'wine':'WineQuality', 'diabetes':'Diabetes'}
    dists = {}; ords = {}
    for key in datas:
        dists[key] = pd.read_csv('./res/sens/{}_res_dist_sens.csv'.format(key)).mean()
        ords[key] = pd.read_csv('./res/sens/{}_res_ord_sens.csv'.format(key)).mean()
    gammas = [10**i for i in range(-3,3)]

    markers = {'fico':'o', 'german':'s', 'wine':'^', 'diabetes':'v'}
    tlps = ['TLPS_ORDER_{}'.format(g) for g in gammas]
    dace = ['DACE_ORDER_{}'.format(g) for g in gammas]
    fig = plt.figure()

    ax1 = fig.add_subplot(3, 1, 1)
    data = 'wine'
    ln1 = ax1.plot(gammas, dists[data][tlps], marker='o', color='blue', label=r'$C_\mathrm{dist}$ (TLPS)')
    ax2 = ax1.twinx()
    ln2 = ax2.plot(gammas, ords[data][tlps], marker='^', linestyle='dashed', color='red', label=r'$C_\mathrm{ord}$')
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1+h2, l1+l2, loc='upper left', borderaxespad=1.2, fontsize=12)
    ax1.set_xlabel(r'$\lambda$', fontsize=16, labelpad=-0.3)
    ax1.set_xscale('log')
    ax1.tick_params(labelsize=12)
    ax2.tick_params(labelsize=12)
    ax1.set_ylabel(r'$C_\mathrm{dist}$', fontsize=16)
    ax2.set_ylabel(r'$C_\mathrm{ord}$', fontsize=16)
    ax1.set_title(DATAS[data], fontsize=16)

    ax1 = fig.add_subplot(3, 1, 2)
    data = 'german'
    ln1 = ax1.plot(gammas, dists[data][dace], marker='o', color='blue', label=r'$C_\mathrm{dist}$ (DACE)')
    ax2 = ax1.twinx()
    ln2 = ax2.plot(gammas, ords[data][dace], marker='^', linestyle='dashed', color='red', label=r'$C_\mathrm{ord}$')
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1+h2, l1+l2, loc='upper left', borderaxespad=1.2, fontsize=12)
    ax1.set_xlabel(r'$\lambda$', fontsize=16, labelpad=-0.3)
    ax1.set_xscale('log')
    ax1.tick_params(labelsize=12)
    ax2.tick_params(labelsize=12)
    ax1.set_ylabel(r'$C_\mathrm{dist}$', fontsize=16)
    ax2.set_ylabel(r'$C_\mathrm{ord}$', fontsize=16)
    ax1.set_title(DATAS[data], fontsize=16)


    plt.tight_layout()
    fig.align_labels()
    plt.savefig('dist_ord_real_select_rev.pdf', bbox_inches='tight', pad_inches=0.05)

def plot_sens():
    import pandas as pd
    from matplotlib import pyplot as plt
    plt.rcParams["font.family"] = 'arial'
    plt.rcParams['text.usetex'] = True

    datas = ['fico','german','wine','diabetes']
    DATAS = {'fico':'FICO', 'german':'German', 'wine':'WineQuality', 'diabetes':'Diabetes'}
    dists = {}; ords = {}
    for key in datas:
        dists[key] = pd.read_csv('./res/sens/{}_res_dist_sens.csv'.format(key)).mean()
        ords[key] = pd.read_csv('./res/sens/{}_res_ord_sens.csv'.format(key)).mean()
    gammas = [10**i for i in range(-3,3)]

    markers = {'fico':'o', 'german':'s', 'wine':'^', 'diabetes':'v'}
    tlps = ['TLPS_ORDER_{}'.format(g) for g in gammas]
    dace = ['DACE_ORDER_{}'.format(g) for g in gammas]

    for data in datas:
        fig = plt.figure()

        ax1 = fig.add_subplot(3, 1, 1)
        ln1 = ax1.plot(gammas, dists[data][tlps], marker='o', color='blue', label=r'$C_\mathrm{dist}$ (TLPS)')
        ax2 = ax1.twinx()
        ln2 = ax2.plot(gammas, ords[data][tlps], marker='^', linestyle='dashed', color='red', label=r'$C_\mathrm{ord}$')
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1+h2, l1+l2, loc='upper left', borderaxespad=1.2, fontsize=12)
        ax1.set_xlabel(r'$\lambda$', fontsize=16, labelpad=-0.3)
        ax1.set_xscale('log')
        ax1.tick_params(labelsize=12)
        ax2.tick_params(labelsize=12)
        ax1.set_ylabel(r'$C_\mathrm{dist}$', fontsize=16)
        ax2.set_ylabel(r'$C_\mathrm{ord}$', fontsize=16)
        ax1.set_title('TLPS', fontsize=16)

        ax1 = fig.add_subplot(3, 1, 2)
        ln1 = ax1.plot(gammas, dists[data][dace], marker='o', color='blue', label=r'$C_\mathrm{dist}$ (DACE)')
        ax2 = ax1.twinx()
        ln2 = ax2.plot(gammas, ords[data][dace], marker='^', linestyle='dashed', color='red', label=r'$C_\mathrm{ord}$')
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1+h2, l1+l2, loc='upper left', borderaxespad=1.5, fontsize=12)
        ax1.set_xlabel(r'$\lambda$', fontsize=16, labelpad=-0.3)
        ax1.set_xscale('log')
        ax1.tick_params(labelsize=12)
        ax2.tick_params(labelsize=12)
        ax1.set_ylabel(r'$C_\mathrm{dist}$', fontsize=16)
        ax2.set_ylabel(r'$C_\mathrm{ord}$', fontsize=16)
        ax1.set_title('DACE', fontsize=16)

        plt.tight_layout()
        fig.align_labels()
        plt.savefig('dist_ord_real_{}.pdf'.format(DATAS[data]), bbox_inches='tight', pad_inches=0.05)

def plot_sens_all():
    import pandas as pd
    from matplotlib import pyplot as plt
    plt.rcParams["font.family"] = 'arial'
    plt.rcParams['text.usetex'] = True

    datas = ['fico','german','wine','diabetes']
    DATAS = {'fico':'FICO', 'german':'German', 'wine':'WineQuality', 'diabetes':'Diabetes'}
    costs = ['TLPS', 'DACE', 'MAD', 'SCM']
    dists = {}; ords = {}
    for key in datas:
        dists[key] = pd.concat([pd.read_csv('./res/sens/{}_res_dist_sens.csv'.format(key)).mean(), pd.read_csv('./res/sens/appendix/{}_res_dist_sens.csv'.format(key)).mean()])
        ords[key] = pd.concat([pd.read_csv('./res/sens/{}_res_ord_sens.csv'.format(key)).mean(), pd.read_csv('./res/sens/appendix/{}_res_ord_sens.csv'.format(key)).mean()])
    gammas = [10**i for i in range(-3,3)]

    tlps = ['TLPS_ORDER_{}'.format(g) for g in gammas]
    dace = ['DACE_ORDER_{}'.format(g) for g in gammas]
    mad  = ['MAD_ORDER_{}'.format(g) for g in gammas]
    scm  = ['SCM_ORDER_{}'.format(g) for g in gammas]
    keys = [tlps, dace, mad, scm]

    for cost, key in zip(costs, keys):
        fig = plt.figure(figsize=[7.5,9])
        for i, data in enumerate(datas):
            ax1 = fig.add_subplot(4, 1, i+1)
            ln1 = ax1.plot(gammas, dists[data][key], marker='o', color='blue', label=r'$C_\mathrm{dist}$'+' ({})'.format(cost))
            ax2 = ax1.twinx()
            ln2 = ax2.plot(gammas, ords[data][key], marker='^', linestyle='dashed', color='red', label=r'$C_\mathrm{ord}$')
            h1, l1 = ax1.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            if(i==0): ax1.legend(h1+h2, l1+l2, loc='upper left', borderaxespad=1.2, fontsize=16)
            ax1.set_xlabel(r'$\lambda$', fontsize=20, labelpad=-0.3)
            ax1.set_xscale('log')
            ax1.tick_params(labelsize=14)
            ax2.tick_params(labelsize=14)
            ax1.set_ylabel(r'$C_\mathrm{dist}$', fontsize=20)
            ax2.set_ylabel(r'$C_\mathrm{ord}$', fontsize=20)
            ax1.set_title(DATAS[data], fontsize=20)
        plt.tight_layout()
        fig.align_labels()
        plt.savefig('dist_ord_real_{}.pdf'.format(cost), bbox_inches='tight', pad_inches=0.05)


def real_to_latex():
    import pandas as pd
    datas = ['fico','german','wine','diabetes']
    DATAS = ['FICO', 'German', 'Wine', 'Diabetes']
    clfs = ['lr', 'rf', 'mlp']
    # costs = ['TLPS', 'DACE']
    costs = ['MAD', 'SCM']
    print('obj')
    for cost in costs:
        for data, DATA in zip(datas, DATAS):
            sent = '{}'.format(DATA)
            for clf in clfs:
                dist = pd.read_csv('./res/{0}/appendix/{1}_res_dist_{0}.csv'.format(clf, data))
                order = pd.read_csv('./res/{0}/appendix/{1}_res_ord_{0}.csv'.format(clf, data))
                dist = dist[dist>0]; order = order[order>0]
                obj = dist + order
                sent += ' & {:.3} $\pm$ {:.2} & {:.3} $\pm$ {:.2}'.format(obj[cost].mean(), obj[cost].std(), obj[cost+'_ORDER_1.0'].mean(), obj[cost+'_ORDER_1.0'].std(), )
            sent += ' \\\\'
            print(sent)
    print('ord')
    for cost in costs:
        for data, DATA in zip(datas, DATAS):
            sent = '{}'.format(DATA)
            for clf in clfs:
                order = pd.read_csv('./res/{0}/appendix/{1}_res_ord_{0}.csv'.format(clf, data))
                order = order[order>0]
                sent += ' & {:.3} $\pm$ {:.2} & {:.3} $\pm$ {:.2}'.format(order[cost].mean(), order[cost].std(), order[cost+'_ORDER_1.0'].mean(), order[cost+'_ORDER_1.0'].std(), )
            sent += ' \\\\'
            print(sent)
    print('time')
    for cost in costs:
        for data, DATA in zip(datas, DATAS):
            sent = '{}'.format(DATA)
            for clf in clfs:
                time = pd.read_csv('./res/{0}/appendix/{1}_res_time_{0}.csv'.format(clf, data))
                time = time[time>0]
                sent += ' & {:.3} $\pm$ {:.2} & {:.3} $\pm$ {:.2}'.format(time[cost].mean(), time[cost].std(), time[cost+'_ORDER_1.0'].mean(), time[cost+'_ORDER_1.0'].std(), )
            sent += ' \\\\'
            print(sent)

