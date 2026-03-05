import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix import QMixer
from modules.mixers.noise_mix import QMixer as NoiseQMixer
import torch as th
from torch.optim import RMSprop, adam
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import utils.trajectory_encoder as tra_enc


class QLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger
        self.n_agents = args.n_agents
        self.params = list(mac.parameters())
        self.last_target_update_episode = 0


        self.encoder = tra_enc.Encoder(self._get_input_shape(scheme), feature_dim=64)
        self.autoregressive = tra_enc.AutoregressiveModel(64, 64)  # Adjust the hidden size as needed

        self.dictionary = LearnableDictionary(self.args.n_agents, self.args.rnn_hidden_dim)

        self.tra_params = list(self.encoder.parameters+self.autoregressive.parameters()+self.dictionary.embeddings)
        self.tra_optimiser = adam(params=self.tra_params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)
        self.IAU = IAU(input_dim=65, n_action=scheme["actions"]["vshape"])
        self.params += list(self.IAU.parameters())
        self.mixer = None
        if args.mixer is not None:
            if args.mixer == "vdn":
                self.mixer = VDNMixer()
            elif args.mixer == "qmix":
                self.mixer = NoiseQMixer(args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
            self.params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)

        self.optimiser = adam(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

        self.target_mac = copy.deepcopy(mac)
        self.target_IAU = copy.deepcopy(IAU)

        self.log_stats_t = -self.args.learner_log_interval - 1

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]
        noise = batch["noise"][:, 0].unsqueeze(1).repeat(1,rewards.shape[1],1)

        tra_reps = []
        for t in range(batch.max_seq_length - 1):
            tra_inputs = self._build_inputs(batch, t)
            tra_embedding = self.encoder(tra_inputs)
            tra_rep = self.autoregressive(tra_embedding)
            tra_reps.append(tra_rep)

        tra_reps = th.stack(tra_reps, dim=1)
        dictionary_embeddings = self.dictionary()

        identity_reps = [dictionary_embeddings[i, :] for i in range(dictionary_embeddings.size(0))]

        infonce_loss = compute_infonce_loss(tra_reps, identity_reps)

        self.tra_optimiser.zero_grad()
        infonce_loss.backward()
        grad_norm_tra = th.nn.utils.clip_grad_norm_(self.tra_params, self.args.grad_norm_clip)
        self.tra_optimiser.step()


        # Calculate estimated Q-Values
        mac_out = []
        tra_reps = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            tra_t = self._build_inputs(batch, t)
            tra_e = self.encoder(tra_t)
            tra_rep = self.autoregressive(tra_e.unsqueeze(0))
            tra_reps.append(tra_rep)
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time
        tra_reps = th.stack(tra_reps, dim=1)

        r_c = calculate_entropy(tra_reps, k=3)
        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim

        IAU_inputs = self._get_IAU_input(batch.batch_size, chosen_action_qvals, tra_reps)
        IAU_out = IAU(IAU_inputs) 
        IAU_out_action_qvals = th.gather(IAU_out[:, :-1], dim=3, index=actions)

        # Calculate the Q-Values necessary for the target
        target_mac_out = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs = self.target_mac.forward(batch, t=t)
            target_mac_out.append(target_agent_outs)

        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_mac_out = th.stack(target_mac_out[1:], dim=1)  # Concat across time

        # Mask out unavailable actions
        target_mac_out[avail_actions[:, 1:] == 0] = -9999999  # From OG deepmarl

        # Max over target Q-Values
        if self.args.double_q:
            # Get actions that maximise live Q (for double q-learning)
            mac_out[avail_actions == 0] = -9999999
            cur_max_actions = mac_out[:, 1:].max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]

        target_IAU_inputs = self._get_IAU_input(batch.batch_size, target_max_qvals, tra_reps)
        target_IAU_out = self.target_IAU(target_IAU_inputs)
        target_IAU_out_max_qvals = target_IAU_out.max(dim=3)[0]

        IAUtargets = r_c.view_as(rewards)+ self.args.gamma * (1 - terminated) * target_IAU_out_max_qvals
        td_error_iau = (IAU_out_action_qvals - IAUtargets.detach())

        mask = mask.expand_as(td_error_iau)
        masked_td_error = td_error_iau * mask
        loss_iau = (masked_td_error ** 2).sum() / mask.sum()

        # Mix
        if self.mixer is not None:
            chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1], noise)
            target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:], noise)

        mac_out[avail_actions == 0] = -9999999
        q_softmax_actions = th.nn.functional.softmax(mac_out[:, :-1], dim=3)

        if self.args.hard_qs:
            maxs = th.max(mac_out[:, :-1], dim=3, keepdim=True)[1]
            zeros = th.zeros_like(q_softmax_actions)
            zeros.scatter_(dim=3, index=maxs, value=1)

        # Calculate 1-step Q-Learning targets
        targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals

        # Td-error
        td_error = (chosen_action_qvals - targets.detach())

        mask = mask.expand_as(td_error)

        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask

        # Normal L2 loss, take mean over actual data
        loss = (masked_td_error ** 2).sum() / mask.sum()

        loss = loss + self.args.alpha * loss_iau

        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            self.logger.log_stat("grad_norm_tra", grad_norm_tra, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item()/mask_elems), t_env)
            self.logger.log_stat("q_taken_mean", (chosen_action_qvals * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.log_stats_t = t_env

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        self.target_IAU.load_state(self.IAU)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    # Contrastive loss function
    def contrastive_loss(self, x, y, temperature=0.07):
        x = th.nn.functional.normalize(x, dim=-1)
        y = th.nn.functional.normalize(y, dim=-1)
        logits = th.matmul(x, y.t()) / temperature
        labels = th.arange(x.size(0)).to(x.device)
        loss = th.nn.CrossEntropyLoss()(logits, labels)
        return loss

    def _build_inputs(self, batch, t):

        bs = batch.batch_size
        inputs = []
        inputs.append(batch["obs"][:, t])  # b1av
        inputs = th.cat([x.reshape(bs, self.n_agents, -1) for x in inputs], dim=2)
        return inputs


    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]

        input_shape += scheme["actions_onehot"]["vshape"][0]


        return input_shape

    def _get_IAU_input(self, size, action_values, encoded):
        IAU_inputs = []
        IAU_inputs.append(action_values)
        IAU_inputs.append(encoded)

        IAU_inputs = th.cat([x.reshape(size, self.n_agents, -1) for x in IAU_inputs], dim=3)
        return IAU_inputs

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        self.IAU.cuda()
        self.target_IAU.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))


    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        return input_shape


def calculate_entropy(trajectories, k=5, d=1):

    batch_size = trajectories.size(0)
    dist_matrix = th.cdist(trajectories, trajectories, p=2)
    dist_matrix += th.eye(batch_size, device=trajectories.device) * 1e9

    # Find the k nearest neighbors (excluding itself)
    _, indices = th.topk(dist_matrix, k=k+1, largest=False, dim=1)
    nearest_neighbors = indices[:, 1:]  # exclude the self-distance at index 0

    # Gather the distances of the k nearest neighbors
    neighbor_distances = th.gather(dist_matrix, 1, nearest_neighbors)
    entropy_terms = th.log(d + (1/k) * th.sum(neighbor_distances ** batch_size, dim=1))

    return entropy_terms

class IAU(th.nn.Module):
    def __init__(self, input_dim, n_action):
        super().__init__()
        self.fc1 = th.nn.Linear(input_dim, 64)
        self.fc2 = th.nn.Linear(64, n_action)

    def forward(self, x):
        y = th.nn.ReLU(self.fc1(x))
        y = self.fc2(y)
        return y


class LearnableDictionary(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super(LearnableDictionary, self).__init__()
        self.embeddings = nn.Parameter(th.randn(num_embeddings, embedding_dim))

    def forward(self):
        return self.embeddings



def compute_infonce_loss(c_t, d):

    logits = th.matmul(c_t, d.T)
    exp_logits = th.exp(logits)

    positive_pairs = th.diagonal(exp_logits, dim1=-2, dim2=-1)
    denominator = exp_logits.sum(dim=-1)
    info_nce_loss = -th.log(positive_pairs / denominator).mean()

    return info_nce_loss

