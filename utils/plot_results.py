import matplotlib.pyplot as plt
import pickle
from scipy.ndimage.filters import uniform_filter1d
import time
import numpy as np
import pandas as pd
from itertools import cycle
import argparse
import os

from numpy import genfromtxt
from numpy.random import choice

from datetime import datetime

def calc_run_time(data, mode='full'):
    '''
    Calculates the ellapsed time between starting the algorithm and first reaching the max-vlaue
    ARGUMENTS:
        - data:             dictionary of the running history
        - mode:             full: total length, sample: first max-value
    OUTPUT:
        - total_seconds:    float
    '''
    if mode == 'full':
        time_pos = -1
    if mode == 'sample':
        time_pos = np.argmax(data['avg_ret']) # first occurence of max value

    start = datetime.fromtimestamp(data['time_start'])
    end = datetime.fromtimestamp(data['timestamps'][time_pos])

    return (end-start).total_seconds()

def plot_time(run_times, labels, env_name, seed, title='Time complexity', no_show=False):
    '''
    Plots the total ellapsed time for multiple runs as barchart
    ARGUMENTS:
        - run_times:    list of float, containing total ellapsed time
        - labels:       list of string, algorithm names used as label on the chart
        - env_name:     string, title of the diagram
        - no_show:      boolaen, save figure instead of showing it, default: FALSE
    OUTPUT:
        - None
    '''
    fig = plt.figure(figsize=(8, 6))
    #for label, std_dev in zip(label_list, std_dev_list):
    axis_font = {'fontname':'Arial', 'size':'14'}
    #plt.legend(loc='lower right', prop={'size' : 12})

    plt.bar(labels, run_times, width=0.2)
    
    plt.title(f'{title} in {env_name}',**axis_font)
    plt.ylabel('Duration (in sec)',**axis_font)
    plt.grid(axis='y')

    if no_show:
        fig.savefig('time_%s_%s.png' % (env_name, seed), dpi=fig.dpi, bbox_inches='tight')
    else:
        plt.show()

def plot_robustness(std_dev_list, label_list, env_name, no_show=False):
    '''
    Plotting standard deviation of running histories of trained agentsPpull
    DO NOT USE IT WITH REGULAR TRAINING HISTORY

    ARGUMENTS:
        - std_dev_list:
        - label_list:
        - 
    '''
    fig = plt.figure(figsize=(16, 8))
    # std-devs for a fully trained model
    #for label, std_dev in zip(label_list, std_dev_list):
    axis_font = {'fontname':'Arial', 'size':'14'}  
    plt.boxplot(std_dev_list, labels=label_list)
    
    plt.grid(axis='y')
    plt.title('Robustness', **axis_font)
    plt.ylabel('Standard deviation', **axis_font)

    if no_show:
        fig.savefig('robustness_%s.png' % env_name, dpi=fig.dpi, bbox_inches='tight')
    else:
        plt.show()

def plot_returns(max_return_list, mean_return_list, labels, env_name, seed, no_show=False):
    '''
    Plotting the max and mean average_return of the models.
    ARGUMENTS:
        - max_return_list:      list of floats, max return value
        - mean_return_list:     list of floats, mean return value
        - labels:               list of string, algorithm names used as label on the chart
        - env_name:             string, title of the diagram
        - no_show:              boolaen, save figure instead of showing it, default: FALSE
    OUTPUT:
        - None
    '''
    x = np.arange(len(labels))
    width = 0.35  # the width of the bars
    
    fig, ax = plt.subplots()

    ax.bar(x - width/2, max_return_list, width=width, label='Max avg_return')
    ax.bar(x + width/2, mean_return_list, width=width, label='Mean avg_return')

    axis_font = {'fontname':'Arial', 'size':'14'}

    ax.set_ylabel('Avg return', **axis_font)
    ax.set_title(f'Scores in {env_name}', **axis_font)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, **axis_font)
    #ax.legend(prop={'size' : 12})
    ax.grid(axis='y')

    fig.tight_layout()
    if no_show:
        fig.savefig('returns_%s_%s.png' % (env_name, seed), dpi=fig.dpi, bbox_inches='tight')
    else:
        plt.show()

def multiple_plot(average_vals_list, std_dev_list, traj_list, other_labels, env_name, seed, smoothing_window=1, no_show=False, ignore_std=False, climit=None, extra_lines=None):
    '''
    Plotting multiple history of running of algorithms.
    ARGUMENTS:
        - average_vals_list:    list of lists of floats, running histories of different algorithms at each evaluation
        - std_dev_list:         list of lists of floats, standard deviations of different algorithms at each evaluation
        - traj_list:            list of lists of integers, evaluation steps, if None: 1 to end will be used for x axis
        - other_labels:         list of string, algorithm names used as label on the chart
        - env_name:             string, title of the diagram
        - smooting_window:      integer, size of the window for smoothing the runnin_histories, default: 5
        - no_show:              boolaen, save figure instead of showing it, default: FALSE
        - ignore_std:           boolean, wheater to show standard deviations, default: FALSE
        - climit:               integer, option to limit the history, defualt: None
        - extra_lines:          list of lists of floats, inlcude other running histories as baselines

    Based on:
    Authors Peter Henderson, Riashat Islam, Philip Bachman, Joelle Pineau, Doina Precup, David Meger
    Copied from https://github.com/Breakend/DeepReinforcementLearningThatMatters
    '''
    fig = plt.figure(figsize=(16, 8))
    colors = ["#1f77b4", "#ff7f0e", "#d62728", "#9467bd", "#2ca02c", "#8c564b", "#e377c2", "#bcbd22", "#17becf"]
    linestyles = ['solid', 'dashed', 'dashdot', 'dotted']
    color_index = 0
    ax = plt.subplot() # Defines ax variable by creating an empty plot
    offset = 1

    # Set the tick labels font
    # for label in (ax.get_xticklabels() + ax.get_yticklabels()):
    #     label.set_fontname('Arial')
    #     label.set_fontsize(28)
    if traj_list is None:
        traj_list = [None]*len(average_vals_list)
    limit = climit
    index = 0
    for average_vals, std_dev, label, trajs in zip(average_vals_list, std_dev_list, other_labels[:len(average_vals_list)], traj_list):
        index += 1
        
        if climit is None:
            limit = len(average_vals)

        # If we don't want reward smoothing, set smoothing window to size 1
        rewards_smoothed_1 = uniform_filter1d(average_vals, size=smoothing_window)[:limit]
        std_smoothed_1 = uniform_filter1d(std_dev, size=smoothing_window)[:limit]
        rewards_smoothed_1 = rewards_smoothed_1[:limit]
        std_dev = std_dev[:limit]
       
        if trajs is None:
            # in this case, we just want to use algorithm iterations, so just take the number of things we have.
            trajs = list(range(len(rewards_smoothed_1)))
        else:
            plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
            ax.xaxis.get_offset_text().set_fontsize(14)

        fill_color = colors[color_index]
        color_index += 1

        cum_rwd_1, = plt.plot(trajs, rewards_smoothed_1, label=label, color=fill_color, ls=linestyles[color_index % len(linestyles)])
        offset += 3
        if not ignore_std:
            # uncomment this to use error bars
            #plt.errorbar(trajs[::25 + offset], rewards_smoothed_1[::25 + offset], yerr=std_smoothed_1[::25 + offset], linestyle='None', color=fill_color, capsize=5)
            plt.fill_between(trajs, rewards_smoothed_1 + std_smoothed_1,   rewards_smoothed_1 - std_smoothed_1, alpha=0.3, edgecolor=fill_color, facecolor=fill_color)

    if extra_lines:
        for lin in extra_lines:
            plt.plot(trajs, np.repeat(lin, len(rewards_smoothed_1)), linestyle='-.', color = colors[color_index], linewidth=2.5, label=other_labels[index])
            color_index += 1
            index += 1

    axis_font = {'fontname':'Arial', 'size':'14'}
    plt.legend(loc='lower right', prop={'size' : 12})
    plt.xlabel("Iterations", **axis_font)
    if traj_list is not None and traj_list[0] is not None:
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        ax.xaxis.get_offset_text().set_fontsize(16)
        plt.xlabel("Timesteps", **axis_font)
    else:
        plt.xlabel("Iterations", **axis_font)
    plt.ylabel("Average Return", **axis_font)
    plt.title("%s"% env_name, **axis_font)

    if no_show:
        fig.savefig('history_%s_%s.png' % (env_name, seed), dpi=fig.dpi, bbox_inches='tight')
    else:
        plt.show()

    return fig

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths_to_progress_pickles", default='for_ati/1234', type=str, help="All the pickles associated with the data")
    parser.add_argument("--env_name",default='Acrobot', help= "This is just the title of the plot and the filename.")
    parser.add_argument("--save", action="store_false")
    parser.add_argument("--ignore_std", action="store_true")
    parser.add_argument('--smoothing_window', default=5, type=int, help="Running average to smooth with, default is 1 (i.e. no smoothing)")
    parser.add_argument('--limit', default=None, type=int)
 
    return parser.parse_args()

if __name__ == "__main__":
    args = parse()
    plt.rc('axes', titlesize=14)
    plt.rc('xtick', labelsize=14) 

    avg_rets = []
    std_dev_rets = []
    labels = []
    run_times = []
    sample_eff = []
    max_returns = []
    mean_returns = []
    seeds = []
    trajs = None

    # Read all pickle's into arrays
    for name in os.listdir(args.paths_to_progress_pickles):
        path = os.path.join(args.paths_to_progress_pickles, name)
        if not os.path.isfile(path):
            continue
        if path.split('.')[-1] != 'pickle':
            continue
        data = pickle.load(open(path, "rb"))

        avg_ret = np.array(data["avg_ret"]) # averge return across trials
        std_dev_ret = np.array(data["std_dev"]) # standard error across trials
        
        if 'lambda' in data:
            labels.append(data['name'].upper() + ' with $\lambda=$ {}'.format(data['lambda']))
        else:
            labels.append(data['name'].upper())
        
        avg_rets.append(avg_ret)
        std_dev_rets.append(std_dev_ret)
        sample_eff.append(calc_run_time(data, mode='sample'))
        run_times.append(calc_run_time(data, mode='full'))
        max_returns.append(np.max(data['avg_ret']))
        mean_returns.append(np.average(data['avg_ret']))
        #seeds.append(data['seed'])
    args.env_name=data['env_name']
    print(args.env_name)
    seed = data['seed']
    #multiple_plot(avg_rets, std_dev_rets, trajs, labels, args.env_name, seed, smoothing_window=args.smoothing_window, no_show=args.save, ignore_std=False, climit=args.limit)
    multiple_plot(avg_rets, std_dev_rets, trajs, labels, args.env_name, seed, smoothing_window=args.smoothing_window, no_show=args.save, ignore_std=args.ignore_std, climit=args.limit)
    print(seed)
    print('Run_times', list(zip(labels,run_times)))
    print('Max_ret', list(zip(labels,max_returns)))
    print('Mean_ret',list(zip(labels,mean_returns)))
    plot_time(run_times, labels, args.env_name, seed, no_show=args.save)
    #plot_time(sample_eff, labels, args.env_name, title='Sample efficiency', no_show=args.save)
    plot_returns(max_returns, mean_returns, labels, args.env_name, seed,no_show=args.save)
    # plot_robustness(std_dev_rets, labels, env_name=args.env_name, no_show=args.save)