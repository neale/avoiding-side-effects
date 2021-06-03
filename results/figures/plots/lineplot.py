import csv
import os
import matplotlib.pyplot as plt
from collections import defaultdict
import matplotlib.patches as mpatches
import numpy as np

from glob import glob as glob
import seaborn as sns
sns.set_style("whitegrid", {'axes.grid': True, 'axes.edgecolor': 'black', "xtick.bottom": True, "ytick.left": True})

import warnings
warnings.filterwarnings("ignore")  # we don't care that sns.tsplot is deprecated

NUM_STEPS = 12e6
N_STEPS = 15e6
SMOOTHING = 20 # rolling mean over SMOOTHING entries

colors = sns.color_palette('colorblind', n_colors=5)[::-1]
colors[0], colors[1] = colors[1], colors[0]  # preserve original color scheme
colors[3] = sns.color_palette('colorblind', n_colors=10)[7]  # gray for AUP_proj

names = ['ppo', 'dqn', 'aup', 'aup-p', 'naive']
labels = [r'$\mathtt{PPO}$', r'$\mathtt{DQN}$', r'$\mathtt{AUP}$', r'$\mathtt{AUP}_{\mathtt{proj}}$',
          r'$\mathtt{Naive}$']
condition_dict = {}
for name, ind in zip(names, range(len(names))):
    condition_dict[name] = {'color': colors[ind], 'label': labels[ind]}  # for plotting convenience

task_dict = {'reward': {'name': 'Reward', 'ylim': None},
             'side_effect': {'name': 'Side effect score', 'ylim': None},
             'length': {'name': 'Episode length', 'ylim': 1000},
             'performance': {'name': 'Fraction of obtainable reward', 'ylim': 1}}


def csv_to_tensor(f, val=-1, factor=1):
    data = []
    with open(f) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for i, row in enumerate(csv_reader):
            if 'wall' in row[0] or 'Wall' in row[0]:
                continue
            n = factor * float(row[val])
            data.append(n)
    return np.array(data)


def smooth(y, box_pts):
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode='valid')
    return y_smooth


for level in ['append', 'prune']:  #
    for value in ('side_effect', 'reward', 'performance', 'side_effect', 'length'):
        print('-' * 86 + '\n{}'.format(value))
        for difficulty in ('spawn', 'still-easy', 'still')[::-1]:
            if difficulty != 'still-easy' and level == 'prune':  # we didn't do prune-still or prune-spawn
                continue
            print (level, ', ', difficulty)
            data = {}
            color_patch = []
            for name in names:
                # Find all relevant files

                files = glob(f'{name}/{level}_{difficulty}/batch*/*{value if value != "side_effect" else "effect"}')
                data[name] = {'values': [csv_to_tensor(f)[::50] for f in files],
                              'steps': [csv_to_tensor(f, -2)[::50] for f in files],
                              'files': files}
                color_patch.append(mpatches.Patch(color=condition_dict[name]['color'], label=condition_dict[name]['label']))

            steps = [data[name]['steps'] for name in names]
            values = [data[name]['values'] for name in names]
            files = [data[name]['files'] for name in names]
            for n in names:
                print (n)
                print ([data[n]['steps'][i][-1] for i in range(len(data[n]['steps']))])

            # truncate at 15M steps
            for name, d, lst, path in zip(names, values, steps, files):
                for i, item in enumerate(lst):
                    if item[-1] > 15e6:
                        gt_index = np.where(item > 15e6)[0][0]
                        d[i] = d[i][:gt_index]
                        lst[i] = item[:gt_index]
                    assert len(d[i]) == len(lst[i])

            fig, ax = plt.subplots()
            for name in names:
                min_len = min([len(x) for x in data[name]['values']])  # make sure all arrays equal length
                arr = np.stack([x[:min_len] for x in data[name]['values']])
                smoothed_arr = np.stack([smooth(x, SMOOTHING) for x in arr])
                time = np.linspace(0, N_STEPS, len(smoothed_arr[0]))
                sns.tsplot(data=smoothed_arr, time=time, color=condition_dict[name]['color'], ax=ax)

            plt.title(r'$\mathtt{' + level + '-' + difficulty + '}$' if value != 'side_effect' else ' ', fontsize=17,
                      pad=10, loc='left')
            plt.legend(frameon=True, fancybox=True, ncol=2,
                       prop={'weight': 'bold', 'size': 10.5}, handles=color_patch)

            # x-axis
            ax.set_xticks(np.linspace(0, min_len, 5))  # major ticks every 3M steps
            ax.set_xticks(np.linspace(0, min_len, int(NUM_STEPS / 1e6) + 1), minor=True)
            ax.set_xticklabels([str(i) if i > 0 else '' for i in range(0, int(NUM_STEPS / 1e6) + 1, 3)],
                               fontdict={'fontsize': 16})
            plt.xlabel('Steps, millions', fontsize=15, weight='semibold', labelpad=10)

            # y-axis
            plt.ylabel(task_dict[value]['name'], fontsize=15, labelpad=10, weight='semibold')
            plt.ylim(bottom=0, top=task_dict[value]['ylim'])
            plt.setp(plt.gca().get_yticklabels(), fontsize=16)

            sns.despine()
            plt.show()
            fig.savefig(os.path.join(os.path.dirname(__file__), 'plots', 'line', value + '_' +
                                           level + '_' + difficulty + '_plot.pdf'), bbox_inches='tight')
            plt.close()

