import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from etpo.utils.util import get_gard_norm, huber_loss, mse_loss
from lion_pytorch import Lion

class ETPOTrainer:

    def __init__(self, args, agent, num_agents):

        self.tpdv = dict(dtype=torch.float32, device=torch.device("cuda:0"))
        self.agent = agent

        self.clip_param = args.clip_param
        self.ppo_epoch = args.ppo_epoch
        self.num_mini_batch = args.num_mini_batch
        self.value_loss_coef = args.value_loss_coef
        self.max_grad_norm = args.max_grad_norm       
        self.huber_delta = args.huber_delta
        self._use_max_grad_norm = args.use_max_grad_norm
        self._use_clipped_value_loss = args.use_clipped_value_loss
        self._use_huber_loss = args.use_huber_loss
        self.lr = args.lr
        self.opti_eps = args.opti_eps

        self.beta = args.beta
        self.polyak = 0.995

        # self.critic_optimizer = torch.optim.Adam(params=self.agent.critic.parameters(), lr=self.lr, eps=self.opti_eps)
        # self.policy_optimizer = torch.optim.Adam(params=self.agent.generator.parameters(), lr=self.lr, eps=self.opti_eps)
        self.critic_optimizer = Lion(params=self.agent.critic.parameters(), lr=self.lr)
        self.policy_optimizer = Lion(params=self.agent.generator.parameters(), lr=self.lr)
        

    def cal_token_mask(self, action_tokens_batch):
        pad_token = self.agent.pad_token_id
        token_mask = (action_tokens_batch != pad_token).int()
        return token_mask

    def cal_policy_loss(self, pi_logits_infer, values_infer, token_mask):
        log_pi = F.log_softmax(pi_logits_infer, dim=-1)
        kl = torch.mean(self.beta * log_pi - values_infer.detach(), dim=-1)
        # kl = torch.sum(torch.exp(log_pi) * (self.beta * log_pi - values_infer.detach()), dim=-1)

        policy_loss = (kl * token_mask).sum() / token_mask.sum()
        # policy_loss = kl.mean()
        return policy_loss
        
    
    def cal_value_loss(self, values_infer, return_batch, action_tokens_batch, token_mask):
        action_tokens_batch = action_tokens_batch.unsqueeze(-1)
        action_value_infer = torch.gather(values_infer, dim=-1, index=action_tokens_batch)

        diff = return_batch[:, :1, :] - action_value_infer
        if self._use_huber_loss:
            value_loss = huber_loss(diff, self.huber_delta)
        else:
            value_loss = mse_loss(diff)

        value_loss = (value_loss * token_mask.unsqueeze(-1)).sum() / token_mask.sum()
        # value_loss = value_loss.mean()

        return value_loss

    def errl_update(self, sample):
        """
        Update actor and critic networks.
        :param sample: (Tuple) contains data batch with which to update networks.

        :return value_loss: (torch.Tensor) value function loss.
        :return critic_grad_norm: (torch.Tensor) gradient norm from critic up9date.
        """
        value_preds_batch, return_batch, obs_batch, action_tokens_batch = sample

        value_preds_batch = torch.from_numpy(value_preds_batch).to("cuda")
        return_batch = torch.from_numpy(return_batch).to("cuda")
        action_tokens_batch = torch.from_numpy(action_tokens_batch).to("cuda")
        token_mask = self.cal_token_mask(action_tokens_batch)

        values_infer, pi_logits_infer = self.agent.infer_for_train(np.concatenate(obs_batch), 
                                            action_tokens_batch.reshape(-1, action_tokens_batch.shape[-1]))
        values_infer = values_infer.view(obs_batch.shape[0], -1, *values_infer.shape[-2:])
        pi_logits_infer = pi_logits_infer.view(action_tokens_batch.shape[0], -1, *pi_logits_infer.shape[-2:])

        # policy update
        policy_loss = self.cal_policy_loss(pi_logits_infer, values_infer, token_mask)
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        if self._use_max_grad_norm:
            policy_grad_norm = nn.utils.clip_grad_norm_(self.agent.generator.parameters(), self.max_grad_norm)
        else:
            policy_grad_norm = get_gard_norm(self.agent.generator.parameters())
        self.policy_optimizer.step()
        policy_loss = policy_loss.item()
        
        # critic update
        value_loss = self.cal_value_loss(values_infer, return_batch, action_tokens_batch, token_mask)

        self.critic_optimizer.zero_grad()
        value_loss.backward()
        if self._use_max_grad_norm:
            critic_grad_norm = nn.utils.clip_grad_norm_(self.agent.critic.parameters(), self.max_grad_norm)
        else:
            critic_grad_norm = get_gard_norm(self.agent.critic.parameters())
        self.critic_optimizer.step()
        value_loss = value_loss.item()

        # target network update
        with torch.no_grad():
            for p, p_targ in zip(self.agent.critic.parameters(), self.agent.target_critic.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)

        return value_loss, critic_grad_norm, policy_loss, policy_grad_norm

    def train(self, buffer):
        """
        Perform a training update using minibatch GD.
        :param buffer: (SharedReplayBuffer) buffer containing training data.
        :param update_actor: (bool) whether to update actor network.

        :return train_info: (dict) contains information regarding training update (e.g. loss, grad norms, etc).
        """
        train_info = {}
        train_info['value_loss'] = 0
        train_info['value_grad_norm'] = 0
        train_info['policy_loss'] = 0
        train_info['policy_grad_norm'] = 0

        data_generator = buffer.language_sampler(self.num_mini_batch)
        update_time = 0
        for sample in data_generator:
            value_loss, value_grad_norm, policy_loss, policy_grad_norm = self.errl_update(sample)
            train_info['value_loss'] += value_loss
            train_info['value_grad_norm'] += value_grad_norm
            train_info['policy_loss'] += policy_loss
            train_info['policy_grad_norm'] += policy_grad_norm
            update_time += 1

        for k in train_info.keys():
            train_info[k] /= update_time
 
        return train_info

    def prep_training(self):
        self.agent.train()

    def prep_rollout(self):
        self.agent.eval()
