import os

import torch
import numpy as np

import matplotlib.pyplot as plt

def smooth(x, delta=2):
    n = x.shape[0]
    b = np.zeros((n,))

    for i in range(n):
        b[i] = x[max(0, i - delta) : min(n, i + delta)].mean()
    
    return b

def create_plot_from_dict(pathDict, title, plot_name="plot.png"):

    # make list of lists for create_plot
    names = []
    paths = []
    for k, v in pathDict.items():
        names.append(k)
        paths.append(v)

    create_plot(paths, names, title, plot_name)


def create_plot(plots: dict, title="plot", plot_name="plot.png"):

    # plots is a dict
    # key -> path
    # value -> ["plot title", [name of all experiments in path]]

    items = []
    #iterate through dict
    for name, paths in plots.items():

        # collect data
        stats = []
        for exp in paths:

            data = torch.load(os.path.join(exp, "model.pt"), map_location=torch.device('cpu'))
            data = np.array(data["stats"]["successes"])
            data = smooth(data)
            stats.append(data)
        
        # try to convert them to np.array, does not work if different plots
        # have different length -> then show message and cut plots to min
        try:
            stats = np.array(stats)
        except:
            print("Cut plot for ", name, title)
            min_length = len(stats[0])
            for s in stats:
                if min_length > len(s):
                    min_length = len(s)

            # cutting
            stats_new = []
            for s in stats:
                stats_new.append(s[0:min_length])

            stats = np.array(stats_new)

        # compute mean and standard deviation
        mean = np.mean(stats, axis=0)
        std = np.std(stats, axis=0)

        items.append([name, mean, std])

    # plot
    fig, axs = plt.subplots(1, 1, figsize=(10, 10))

    for name, mean, std in items:
        axs.plot(mean,  linewidth=2.0, label=name)
        std1 = np.clip(mean - std, a_min=0, a_max=1)
        std2 = np.clip(mean + std, a_min=0, a_max=1)
        #std1 = mean - std
        #std2 = mean + std
        axs.fill_between(np.arange(len(mean)), std1, std2,  alpha=0.3)

    axs.set_title(title, fontsize=22)
    axs.set_xlim(0, 400)
    axs.set_xlabel("1e6 environment steps", loc="center", fontsize=18)
    axs.set_xticks(np.arange(0, 440, 40))
    x_ticks = ['{0:.1f}'.format(x) for x in (np.arange(0, 2.2, 0.2) if "Hand" in title else np.arange(0, 1.1, 0.1))]
    axs.set_xticklabels(x_ticks)
    axs.set_ylim(0, 1.05)
    axs.set_yticks(np.arange(0, 1.05, 0.1))
    axs.set_ylabel("success rate", fontsize=18, loc="center")

    axs.tick_params(axis="x", labelsize=16)
    axs.tick_params(axis="y", labelsize=16)

    axs.legend(fontsize=18, loc='upper left')

    # TODO

    axs.grid()

    plt.tight_layout()
    plt.savefig(f"../{plot_name}.png")
    plt.close()

def get_subfolders_as_dict(plots:dict):
    # generate a dictionary with all subfolders
    final = {}
    for k in plots.keys():
        all_exp = os.listdir(k)
        final[plots[k]] = [os.path.join(k, e) for e in all_exp]

    return final

def plot_baselines(env_name="FetchPush"):
    # path to experiments
    path = os.path.join("experiments", "baseline", "ddpg", env_name)

    # name of subfolder : title in plot
    # all experiments in subfolder will be used to generate one plot
    plots = {
        os.path.join(path, "iqe"): "DDPG IQE + HER",
        os.path.join(path, "iqe_sym"): "DDPG IME + HER",
        os.path.join(path, "monolithic"): "DDPG monolithic + HER"
    }

    return get_subfolders_as_dict(plots)

def plot_pher(env_name="FetchPush"):
    # path to experiments
    path = os.path.join("experiments", "baseline", "ddpg", env_name)

    # name of subfolder : title in plot
    # all experiments in subfolder will be used to generate one plot
    plots = {
        os.path.join(path, "monolithic"): "DDPG + monolithic + HER",
        os.path.join(path, "monolithic_pher2"): "DDPG + monolithic + new sampling"
    }

    return get_subfolders_as_dict(plots)

def plot_hyperparam_search(env_name="FetchPush", critic="iqe"):
    # path to experiments
    path = os.path.join("experiments", "tuning", "ddpg", env_name)

    # name of subfolder : title in plot
    # all experiments in subfolder will be used to generate one plot
    plots = {
        os.path.join(path, f"{critic}_0.001"): f"DDPG {critic} lr0.001",
        os.path.join(path, f"{critic}_0.0003"): f"DDPG {critic} lr0.0003",
        os.path.join(path, f"{critic}_0.0005"): f"DDPG {critic} lr0.0005",
        os.path.join(path, f"{critic}_0.0001"): f"DDPG {critic} lr0.0001"
    }

    return get_subfolders_as_dict(plots)

if __name__ == "__main__":

    # plot baselines
    #names = [
    #    "FetchPush", "FetchPick", "FetchSlide", 
    #    "HandManipulateBlockRotateParallel", "HandManipulateBlockRotateXYZ", "HandManipulateBlockFull",
    #    "HandManipulateEggFull", "HandManipulateEggRotate",
    #    "HandManipulatePenFull", "HandManipulatePenRotate"
    #]
    names = [
        #"FetchPush", "FetchPick", "FetchSlide",
        #"HandManipulateBlockRotateZ",
        "HandManipulateBlockRotateParallel", 
        "HandManipulateBlockRotateXYZ", 
        "HandManipulateBlockFull",
        #"HandManipulateEggRotate", 
        "HandManipulateEggFull",
        "HandManipulatePenFull", 
        "HandManipulatePenRotate"
    ]
    for name in names:
        #plots = plot_baselines(name)
        plots = plot_pher(name)

        create_plot(plots, title=name, plot_name=f"imgs/exp2/monolithic/ddpg_{name}")

    '''# plot learning rate tuning
    names = [
        ("FetchPick", "iqe"), 
        ("FetchPick", "monolithic"), 
        ("FetchPush", "iqe"),
        ("FetchPush", "monolithic"),
        ("HandManipulateBlockRotateZ", "iqe"),
        ("HandManipulateBlockRotateZ", "monolithic")
    ]

    all_plots = {
        "lr0.001": list(),
        "lr0.0003": list(),
        "lr0.0005": list(),
        "lr0.0001": list()
    }
    for name, critic in names:
        plots = plot_hyperparam_search(name, critic)

        for k, v in plots.items():
            if "lr0.001" in k:
                all_plots["lr0.001"].extend(v)
            if "lr0.0003" in k:
                all_plots["lr0.0003"].extend(v)
            if "lr0.0005" in k:
                all_plots["lr0.0005"].extend(v)
            if "lr0.0001" in k:
                all_plots["lr0.0001"].extend(v)

        create_plot(plots, title=name, plot_name=f"ddpg_tuning_{name}_{critic}")
    create_plot(all_plots, title="Cumulated performance", plot_name=f"ddpg_tuning_all")'''
