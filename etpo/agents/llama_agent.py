from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch
import numpy as np
# from mappo.utils.util import update_linear_schedule
from transformers import LlamaForCausalLM
import copy


class LlamaAgent:

    def __init__(self, model_name):
        # model = "/data/models/muning/CodeLlama-7b-hf"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, padding_side='left')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.generator = AutoModelForCausalLM.from_pretrained(model_name, 
                                                              torch_dtype=torch.float16,
                                                              device_map="auto")
        self.max_new_tokens = 256
        self.pad_token_id = self.tokenizer.bos_token_id

        self.critic = copy.deepcopy(self.generator)
        self.original_policy = copy.deepcopy(self.generator)
        for p in self.original_policy.parameters():
            p.requires_grad = False

        self.target_critic = copy.deepcopy(self.generator)
        for p in self.target_critic.parameters():
            p.requires_grad = False

    def get_actions(self, obs):
        """
        Compute actions and value function predictions for the given inputs.
        """
        prompts = obs.tolist()
        token_seq = self.tokenizer(prompts, return_tensors="pt", padding=True)
        input_ids = token_seq["input_ids"].to("cuda")
        attn_mask = token_seq["attention_mask"].to("cuda")
        # print("input_ids shape: ", input_ids.shape)
        output = self.generator.generate(
            input_ids,
            attention_mask=attn_mask,
            do_sample=True,
            top_k=50,
            temperature=0.5,
            max_new_tokens=self.max_new_tokens,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.pad_token_id,
            return_dict_in_generate=True,
        )
        sequences = output.sequences.to("cpu")

        actions = []
        action_tokens = np.ones(shape=(sequences.shape[0], self.max_new_tokens), dtype=np.int64) * self.pad_token_id
        for i in range(sequences.shape[0]):
            action_token = sequences[i][input_ids[i].shape[0]:]
            action_tokens[i][:action_token.shape[0]] = action_token
            action = self.tokenizer.decode(action_token, skip_special_tokens=True)
            actions.append(action)
        actions = np.array(actions, dtype=np.object_)

        # value function
        critic_input_ids = torch.cat((input_ids, torch.from_numpy(action_tokens).to("cuda")), dim=-1)
        critic_attn_mask = torch.ones_like(critic_input_ids).to("cuda")
        critic_attn_mask[:, :attn_mask.shape[1]] = attn_mask
        # critic_output = self.critic(
        critic_output = self.target_critic(
            input_ids=critic_input_ids,
            attention_mask=critic_attn_mask,
            return_dict=True,
        )
        # the second last output is the logits of the last action token
        values = critic_output.logits[:, -self.max_new_tokens-1:-1, :]
        values = values.detach().cpu().numpy()

        # pi_logit
        pi_input_ids = critic_input_ids
        pi_attn_mask = critic_attn_mask
        pi_output = self.generator(
            input_ids=pi_input_ids,
            attention_mask=pi_attn_mask,
            return_dict=True,
        )
        pi_logits = pi_output.logits[:, -self.max_new_tokens-1:-1, :]
        # print("pi_logits shape: ", pi_logits.shape)
        pi_logits = pi_logits.detach().cpu().numpy()

        # rho_logits
        rho_input_ids = critic_input_ids
        rho_attn_mask = critic_attn_mask
        rho_output = self.original_policy(
            input_ids=rho_input_ids,
            attention_mask=rho_attn_mask,
            return_dict=True,
        )
        rho_logits = rho_output.logits[:, -self.max_new_tokens-1:-1, :]
        # print("rho_logits shape: ", rho_logits.shape)
        rho_logits = rho_logits.detach().cpu().numpy()

        return actions, action_tokens, values, pi_logits, rho_logits

    def get_next_values(self, obs):
        """
        Get value function predictions.
        """
        prompts = obs.tolist()
        token_seq = self.tokenizer(prompts, return_tensors="pt", padding=True)
        input_ids = token_seq["input_ids"].to("cuda")
        attn_mask = token_seq["attention_mask"].to("cuda")

        # q values
        # critic_output = self.critic(
        critic_output = self.target_critic(
            input_ids=input_ids,
            attention_mask=attn_mask,
            return_dict=True,
        )
        values = critic_output.logits[:, -1, :]
        values = values.detach().cpu().numpy()

        # pi logits
        pi_output = self.generator(
            input_ids=input_ids,
            attention_mask=attn_mask,
            return_dict=True,
        )
        pi_logits = pi_output.logits[:, -1, :]
        pi_logits = pi_logits.detach().cpu().numpy()

        # rho logits
        rho_output = self.original_policy(
            input_ids=input_ids,
            attention_mask=attn_mask,
            return_dict=True,
        )
        rho_logits = rho_output.logits[:, -1, :]
        rho_logits = rho_logits.detach().cpu().numpy()

        return values, pi_logits, rho_logits
    
    def infer_for_train(self, obs, action_tokens):
        prompts = obs.tolist()
        token_seq = self.tokenizer(prompts, return_tensors="pt", padding=True)
        input_ids = token_seq["input_ids"].to("cuda")
        attn_mask = token_seq["attention_mask"].to("cuda")

        join_input_ids = torch.cat((input_ids, action_tokens), dim=-1)
        join_attn_mask = torch.ones_like(join_input_ids).to("cuda")
        join_attn_mask[:, :attn_mask.shape[1]] = attn_mask

        critic_output = self.critic(
            input_ids=join_input_ids,
            attention_mask=join_attn_mask,
            return_dict=True,
        )
        values_infer = critic_output.logits[:, -self.max_new_tokens-1:-1, :]

        pi_output = self.generator(
            input_ids=join_input_ids,
            attention_mask=join_attn_mask,
            return_dict=True,
        )
        pi_logits_infer = pi_output.logits[:, -self.max_new_tokens-1:-1, :]

        return values_infer, pi_logits_infer

    def save(self, save_dir, episode):
        torch.save(self.generator.state_dict(), str(save_dir) + "/model_" + str(episode) + ".pt")

    def restore(self, save_dir, episode):
        state_dict = torch.load(str(save_dir) + "/model_" + str(episode) + ".pt")
        self.generator.load_state_dict(state_dict)

    def train(self):
        self.generator.train()
        self.critic.train()

    def eval(self):
        self.generator.eval()
        self.critic.eval()

