import joblib
import torch
import uuid
from rlkit.envs.wrappers import TimeLimit
filename = str(uuid.uuid4())
import numpy as np
import os, itertools, pickle
import os.path as osp
import json
import matplotlib.pyplot as plt


def load_pkl(args, discretized=False):
    file = args.file
    if discretized:
        policy_and_relabelers = []
        folders = list(os.walk(file))[0][1]
        for folder in folders:
            jsonpath = osp.join(file, folder, 'variant.json')
            with open(jsonpath, "r") as read_file:
                json_data = json.load(read_file)
            #     idx = json_data['deterministic']
            data = torch.load(osp.join(file, folder, 'itr_250.pkl'))
            # data = joblib.load(osp.join(file, folder, 'itr_250.pkl'))
            policy = data['evaluation/policy']
            env = data['evaluation/env']
            if isinstance(env, TimeLimit):
                print("prev time limit:", env._max_episode_steps)
                env._max_episode_steps = args.H
                print('changed time limit to:', env._max_episode_steps)
                env.wrapped_env.wrapped_env.energy_factor = args.energy_factor
            print("Policy loaded")
            relabeler = data['replay_buffer/relabeler']
            policy_and_relabelers.append([policy, relabeler])
        relabeler._wrapped_relabeler.n_sampled_latents = 100
    else:
        data = torch.load(file)
        policy = data['evaluation/policy']
        env = data['evaluation/env']
        if isinstance(env, TimeLimit):
            print("prev time limit:", env._max_episode_steps)
            env._max_episode_steps = args.H
            print('changed time limit to:', env._max_episode_steps)
            # env.wrapped_env.wrapped_env.energy_factor = args.energy_factor
        print("Policy loaded")
        relabeler = data['replay_buffer/relabeler']

        jsonpath = osp.join(osp.dirname(file), 'variant.json')
        with open(jsonpath, "r") as read_file:
            json_data = json.load(read_file)

    if json_data['test']:
        print('test was ', relabeler.test)
        relabeler.test = True
    if discretized:
        return policy_and_relabelers, env, json_data
    else:
        return policy, env, relabeler, json_data


def make_transition_matrix(tm, classes,
                           normalize=False,
                           title='Confusion matrix',
                           cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        tm = tm.astype('float') / tm.sum(axis=1)[:, np.newaxis]
        print("Normalized transition matrix")
    else:
        print('Transition matrix, without normalization')

    print(tm)

    plt.imshow(tm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    # fmt = '.2f' if normalize else 'd'
    fmt = '.2f' if normalize else '.2f'
    thresh = tm.max() / 2.
    for i, j in itertools.product(range(tm.shape[0]), range(tm.shape[1])):
        plt.text(j, i, format(tm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if tm[i, j] > thresh else "black")

    plt.ylabel('Original Latent')
    plt.xlabel('Relabeled Latent')
    plt.tight_layout()
    plt.savefig("plots/{}.png".format(title))


def variant_to_name(variant):
    '''
    :param variant: dict created from the variant.json
    :return: string with the understood name of the method used
    '''

    # possible return values: none, random, rew, adv, irl with rew, irl with adv
    if not variant['relabel']:
        return "no relabeling"
    elif variant['approx_irl'] and variant['use_advantages']:
        return "irl with advantage tiebreak"
    elif variant['approx_irl'] and not variant['use_advantages']:
        return "irl with reward tiebreak"
    elif not variant['approx_irl'] and variant['use_advantages']:
        return "advantage relabeling"
    elif not variant['approx_irl'] and not variant['use_advantages']:
        return "reward relabeling"
    else:
        raise RuntimeError("variant not understood")



