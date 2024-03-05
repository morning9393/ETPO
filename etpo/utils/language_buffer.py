import torch
import numpy as np
import torch.nn.functional as F
from etpo.utils.util import get_shape_from_obs_space, get_shape_from_act_space


def _flatten(T, N, x):
    return x.reshape(T * N, *x.shape[2:])


def _cast(x):
    return x.transpose(1, 2, 0, 3).reshape(-1, *x.shape[3:])


def _shuffle_agent_grid(x, y):
    rows = np.indices((x, y))[0]
    # cols = np.stack([np.random.permutation(y) for _ in range(x)])
    cols = np.stack([np.arange(y) for _ in range(x)])
    return rows, cols


class LanguageBuffer(object):
    """
    Buffer to store training data.
    :param args: (argparse.Namespace) arguments containing relevant model, policy, and env information.
    :param num_agents: (int) number of agents in the env.
    """

    def __init__(self, args, num_agents):
        self.episode_length = args.episode_length
        self.n_rollout_threads = args.n_rollout_threads
        self.gamma = args.gamma
        self.gae_lambda = args.gae_lambda
        self._use_gae = args.use_gae
        self._use_popart = args.use_popart
        self._use_valuenorm = args.use_valuenorm
        self.algo = args.algorithm_name
        self.num_agents = num_agents

        self.max_new_tokens = 256
        self.vacab_size = 32016
        self.pad_token_id = 1
        self.beta = args.beta
        # when max_batch = 1, this is an on-policy buffer, otherwise it is a replaybuffer
        self.max_batch = 1
        
        self.cur_num_batch = 0
        self.cur_batch_index = 0
        self.pre_batch_index = None

        self.obs = np.empty((self.max_batch, self.episode_length + 1, self.n_rollout_threads, num_agents), dtype=np.object_)
        self.actions = np.empty((self.max_batch, self.episode_length, self.n_rollout_threads, num_agents), dtype=np.object_)
        self.action_tokens = np.empty((self.max_batch, self.episode_length, self.n_rollout_threads, num_agents, self.max_new_tokens), dtype=np.int64)
        self.value_preds = np.zeros(
            (self.max_batch, self.episode_length + 1, self.n_rollout_threads, num_agents, self.max_new_tokens, self.vacab_size), dtype=np.float32)
        self.returns = np.zeros(
            (self.max_batch, self.episode_length, self.n_rollout_threads, num_agents, self.max_new_tokens, 1), dtype=np.float32)
        self.rewards = np.zeros(
            (self.max_batch, self.episode_length, self.n_rollout_threads, num_agents, 1), dtype=np.float32)
        self.masks = np.ones((self.max_batch, self.episode_length + 1, self.n_rollout_threads, num_agents, 1), dtype=np.float32)
        self.pi_logits = np.zeros((self.max_batch, self.episode_length + 1, self.n_rollout_threads, num_agents, self.max_new_tokens, self.vacab_size), dtype=np.float32)
        self.rho_logits = np.zeros((self.max_batch, self.episode_length + 1, self.n_rollout_threads, num_agents, self.max_new_tokens, self.vacab_size), dtype=np.float32)
        if self.algo == "PPO":
            self.advantages = np.zeros_like(self.returns)
        
        self.step = 0


    def insert(self, obs, actions, value_preds, rewards, masks, action_tokens, pi_logits, rho_logits):
        """
        Insert data into the buffer.
        """
        self.obs[self.cur_batch_index, self.step + 1] = obs.copy()
        self.actions[self.cur_batch_index, self.step] = actions.copy()
        self.value_preds[self.cur_batch_index, self.step] = value_preds.copy()
        self.rewards[self.cur_batch_index, self.step] = rewards.copy()
        self.masks[self.cur_batch_index, self.step + 1] = masks.copy()
        self.action_tokens[self.cur_batch_index, self.step] = action_tokens.copy()
        self.pi_logits[self.cur_batch_index, self.step] = pi_logits.copy()
        self.rho_logits[self.cur_batch_index, self.step] = rho_logits.copy()

        self.step = (self.step + 1) % self.episode_length        

    def after_update(self):
        """Copy last timestep data to first index. Called after update to model."""
        self.pre_batch_index = self.cur_batch_index
        self.cur_batch_index = (self.cur_batch_index + 1) % self.max_batch
        self.obs[self.cur_batch_index, 0] = self.obs[self.pre_batch_index, -1].copy()

    @torch.no_grad()
    def kl_cal(self, pi_logits, rho_logits, values):
        pi_logits = torch.from_numpy(pi_logits).to("cuda")
        rho_logits = torch.from_numpy(rho_logits).to("cuda")
        values = torch.from_numpy(values).to("cuda")

        pi = F.softmax(pi_logits, dim=-1)
        rho = F.softmax(rho_logits, dim=-1)
        kl = torch.sum(pi * (torch.log(pi) - torch.log(rho)), dim=-1, keepdim=True)
        kl = kl.cpu().numpy()

        expected_values = torch.sum(pi * values, dim=-1, keepdim=True)
        expected_values = expected_values.cpu().numpy()
        return kl, expected_values
    
    def get_last_token_position(self, action_tokens):
        pos = len(action_tokens) - 1
        while action_tokens[pos] == self.pad_token_id:
            pos -= 1
        return pos
    
    @torch.no_grad()
    def v_cal(self, pi_logits, rho_logits, values):
        pi_logits = torch.from_numpy(pi_logits).to("cuda")
        rho_logits = torch.from_numpy(rho_logits).to("cuda")
        values = torch.from_numpy(values).to("cuda")
        
        pi = F.softmax(pi_logits, dim=-1)
        rho = F.softmax(rho_logits, dim=-1)
        approx_kl = torch.mean(torch.log(pi) - torch.log(rho), dim=-1)
        approx_kl = torch.mean(approx_kl, dim=-1, keepdim=True)
        approx_kl = approx_kl.cpu().numpy()
        
        v_values = torch.sum(pi * values, dim=-1, keepdim=True)
        # only the first token is valid for state value
        v_values = v_values[:, :, :, 0, :]
        v_values = v_values.cpu().numpy()
        return v_values, approx_kl
        

    def batch_process(self, next_value, next_pi_logits, next_rho_logits):
        """
        Compute returns either as discounted sum of rewards, or using GAE.
        :param next_value: (np.ndarray) value predictions for the step after the last episode step.
        """
        self.value_preds[self.cur_batch_index, -1, :, :, 0, :] = next_value
        self.pi_logits[self.cur_batch_index, -1, :, :, 0, :] = next_pi_logits
        self.rho_logits[self.cur_batch_index, -1, :, :, 0, :] = next_rho_logits
        
        kl_div, mean_values = self.kl_cal(self.pi_logits[self.cur_batch_index], self.rho_logits[self.cur_batch_index], self.value_preds[self.cur_batch_index])
        for step in reversed(range(self.episode_length)):
            for thread in range(self.n_rollout_threads):
                last_token = self.get_last_token_position(self.action_tokens[self.cur_batch_index, step, thread, 0, :])
                for token in reversed(range(self.max_new_tokens)):
                    if token == self.max_new_tokens - 1 or token == last_token:
                        self.returns[self.cur_batch_index, step, thread, :, token, :] = self.rewards[self.cur_batch_index, step, thread, :, :] \
                            + self.gamma * (mean_values[step + 1, thread, :, 0, :] - self.beta * kl_div[step + 1, thread, :, 0, :]) * self.masks[self.cur_batch_index, step + 1, thread, :, :]
                    else:
                        self.returns[self.cur_batch_index, step, thread, :, token, :] = mean_values[step, thread, :, token + 1, :] - self.beta * kl_div[step + 1, thread, :, token + 1, :]
                        # self.returns[self.cur_batch_index, step, thread, :, token, :] = self.gamma * (mean_values[step, thread, :, token + 1, :] - self.beta * kl_div[step + 1, thread, :, token + 1, :])
        
        self.cur_num_batch = self.cur_num_batch + 1 if self.cur_num_batch < self.max_batch else self.max_batch


    def language_sampler(self, num_mini_batch=None, mini_batch_size=None):
        """
        Yield training data for ERRL.
        :param num_mini_batch: (int) number of minibatches to split the batch into.
        :param mini_batch_size: (int) number of samples in each minibatch.
        """
        batch_size = self.n_rollout_threads * self.episode_length * self.cur_num_batch
        # num_mini_batch is the number of mini batches to split per single batch into thus should multiply cur_num_batch
        num_mini_batch *= self.cur_num_batch

        if mini_batch_size is None:
            assert batch_size >= num_mini_batch
            mini_batch_size = batch_size // num_mini_batch

        rand = torch.randperm(batch_size).numpy()
        sampler = [rand[i * mini_batch_size:(i + 1) * mini_batch_size] for i in range(num_mini_batch)]
        rows, cols = _shuffle_agent_grid(batch_size, self.num_agents)

        # keep (num_agent, dim)
        value_preds = self.value_preds[:, :-1].reshape(-1, *self.value_preds.shape[3:])
        value_preds = value_preds[rows, cols]
        returns = self.returns.reshape(-1, *self.returns.shape[3:])
        returns = returns[rows, cols]
        obs = self.obs[:, :-1].reshape(-1, *self.obs.shape[3:])
        obs = obs[rows, cols]
        action_tokens = self.action_tokens.reshape(-1, *self.action_tokens.shape[3:])
        action_tokens = action_tokens[rows, cols]
        if self.algo == "PPO":
            advantages = self.advantages.reshape(-1, *self.advantages.shape[3:])
            advantages = advantages[rows, cols]
            pi_logits = self.pi_logits[:, :-1].reshape(-1, *self.pi_logits.shape[3:])
            pi_logits = pi_logits[rows, cols]

        for indices in sampler:
            # [L,T,N,Dim]-->[L*T,N,Dim]-->[index,N,Dim]-->[index*N, Dim]
            # value_preds_batch = value_preds[indices].reshape(-1, *value_preds.shape[2:])
            # return_batch = returns[indices].reshape(-1, *returns.shape[2:])
            # o_a_embd_batch = o_a_embds[indices].reshape(-1, *o_a_embds.shape[2:])
            value_preds_batch = value_preds[indices]
            return_batch = returns[indices]
            obs_batch = obs[indices]
            action_tokens_batch = action_tokens[indices]

            if self.algo == "PPO":
                advantages_batch = advantages[indices]
                pi_logits_batch = pi_logits[indices]
                yield pi_logits_batch, value_preds_batch, return_batch, obs_batch, action_tokens_batch, advantages_batch
            else:
                yield value_preds_batch, return_batch, obs_batch, action_tokens_batch


