#!/usr/bin/env python
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3,4,5,6,7"
import socket
import setproctitle
import numpy as np
from pathlib import Path
import torch
sys.path.append("../../")
from etpo.config import get_config
from etpo.envs.datascience.scikit_env import ScikitEnv
from etpo.envs.llm_env_wrappers import ShareSubprocVecEnv
from etpo.runner.shared.llm_agent_runner import LLMAgentRunner as Runner

def make_train_env(all_args):
    def get_env_fn(rank):
        def init_env():
            env = ScikitEnv(mode="train", rank=rank, dataset_name=all_args.dataset_name, split=all_args.split)
            env.seed(all_args.seed + rank * 1000)
            return env

        return init_env
    return ShareSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


def parse_args(args, parser):
    all_args = parser.parse_known_args(args)[0]
    return all_args

def build_run_dir(all_args):
    run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[
                       0] + "/scripts/results") / all_args.experiment_name / all_args.algorithm_name / all_args.dataset_name
    if not run_dir.exists():
        os.makedirs(str(run_dir))
        curr_run = 'run1'
    else:
        exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in run_dir.iterdir() if
                            str(folder.name).startswith('run')]
        if len(exst_run_nums) == 0:
            curr_run = 'run1'
        else:
            curr_run = 'run%i' % (max(exst_run_nums) + 1)
    run_dir = run_dir / curr_run
    if not run_dir.exists():
        os.makedirs(str(run_dir))
    return run_dir

def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)
    print("algorithm: {}, dataset_name: {}".format(all_args.algorithm_name, all_args.dataset_name))
    run_dir = build_run_dir(all_args)

    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    envs = make_train_env(all_args)

    config = {
        "all_args": all_args,
        "envs": envs,
        "num_agents": envs.n_agents,
        "run_dir": run_dir
    }

    runner = Runner(config)
    runner.run()

    # post process
    if envs is not None:
        envs.close()

    runner.writter.export_scalars_to_json(str(runner.log_dir + '/summary.json'))
    runner.writter.close()


if __name__ == "__main__":
    main(sys.argv[1:])
