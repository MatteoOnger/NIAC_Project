import logging
import math
import numpy as np
import random
import scallopy
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm

from typing import *

from ..pacman.arena import AvoidingArena
from .utils import extract_cell, Memory, Transition



LOGGER = logging.getLogger(__name__)



class PolicyNet(nn.Module):
    """
    """

    def __ini__(self):
        #TODO:
        pass


    def forward():
        #TODO: tutto
        pass


    def _logical_comp(self, agent_p :torch.Tensor, target_p :torch.Tensor, enemy_p :torch.Tensor) -> torch.Tensor:
        """
        """
        ctx = scallopy.ScallopContext(provenance=self.provenance)

        # create relations
        ctx.add_relation("actor", (int, int))
        ctx.add_relation("target", (int, int))
        ctx.add_relation("node", (int, int))

        # add facts to relations
        ctx.add_facts("actor", [(agent_p[node], node) for node in self.nodes])
        ctx.add_facts("target", [(target_p[node], node) for node in self.nodes])
        ctx.add_facts(
            "node",
            [(torch.clip(1 - enemy_p[node] - self.step_cost, min=0), node) for node in self.nodes],
        ) # <- self.step_cost> to penalise longer walks

        # rules defining edges according to valid moves
        ctx.add_rule(f"edge(x, y, xp, y, {ACTION.RIGHT.value}) = node(x, y) and node(xp, y) and xp == x + 1")
        ctx.add_rule(f"edge(x, y, x, yp, {ACTION.UP.value})    = node(x, y) and node(x, yp) and yp == y + 1")
        ctx.add_rule(f"edge(x, y, xp, y, {ACTION.LEFT.value})  = node(x, y) and node(xp, y) and xp == x - 1")
        ctx.add_rule(f"edge(x, y, x, yp, {ACTION.DOWN.value})  = node(x, y) and node(x, yp) and yp == y - 1")
        
        # compute paths
        ctx.add_rule("path(x, y, x, y) = node(x, y)")
        ctx.add_rule("path(x, y, xp, yp) = edge(x, y, xp, yp, _)")
        ctx.add_rule("path(x, y, xpp, ypp) = path(x, y, xp, yp) and edge(xp, yp, xpp, ypp, _)")

        # next moves
        ctx.add_rule("next_position(xp, yp, a) = actor(x, y) and edge(x, y, xp, yp, a)")
        ctx.add_rule("next_action(a) = next_position(x, y, a) and path(x, y, gx, gy) and target(gx, gy)")

        ctx.run()

        # get results
        res = list(ctx.relation("next_action"))
        actions = [ACTION(res[i][1][0]) for i in range(len(res))]
        probs = torch.tensor([res[i][0] for i in range(len(res))])

        tmp = {actions[i].name:round(float(probs[i]), 2) for i in range(len(res))}

        LOGGER.debug(
            f"positions:\n"
            f" - agent -> {self.nodes[torch.argmax(agent_p)]} with prob. {torch.max(agent_p)}\n"
            f" - target -> {self.nodes[torch.argmax(target_p)]} with prob. {torch.max(target_p)}\n"
            f" - enemies -> {[node for node in self.nodes if enemy_p[node] > 0.5]} with prob. {enemy_p[enemy_p > 0.5].tolist()}"
        )
        LOGGER.info(f"next action: {tmp}")
        return actions[torch.argmax(probs)]


    def _neural_comp(self, rgb_array :np.ndarray) -> Dict[str:torch.Tensor]:
        cells = [extract_cell(rgb_array, node[0], node[1], self.cell_size) for node in self.nodes]
        #TODO: tutto il resto...
        pass



class Agent():
    """
    """

    def __init__(
        self,
        arena :AvoidingArena,
        num_epochs :int,
        num_episodes :int,
        batch_size :int,
        memory_size :int,
        gamma :float=0.99,
        eps_start :float=0.9,
        eps_end :float=0.05,
        eps_decay :float=1000,
        lr :float=1e-4,
        tau :float=5e-3,
        provenance :str='difftopkproofs',
        step_cost :float=0.1
    ):
        """
        """
        self.arena = arena

        self.num_epochs = num_epochs
        self.num_episodes = num_episodes

        self.batch_size = batch_size
        self.memory_size = memory_size

        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.lr = lr
        self.tau = tau
        
        self.provenance = provenance
        self.step_cost = step_cost

        self.memory = Memory(self.memory_size)
        self.nodes = [(i,j) for i in range(self.grid_x) for j in range(self.grid_y)]

        #TODO: quanto segue
        #self.policy_net = ....
        #self.target_net = ....
        
        #self.target_net.load_state_dict(
        # self.policy_net.state_dict()
        #)

        self.criterion = nn.SmoothL1Loss()
        self.optimizer = optim.RMSprop(self.policy_net.parameters(), self.lr)
        return


    def select_action(self, rgb_array :np.ndarray) -> AvoidingArena.Actions:
        """
        """
        action_scores, _ = self.policy_net(rgb_array)
        action = torch.argmax(action_scores, dim=1)
        return action


    def _optimize(self):
        """
        """
        if len(self.memory) < self.batch_size:
            LOGGER.warning("not enough transactions in memory")
            return None
        
        # draw random transitions
        transitions = self.memory.sample(self.batch_size)
        # from batch-array of transitions to transition of batch-arrays
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor([not tr.done for tr in batch], dtype=torch.bool)
        non_final_next_states = torch.cat([ns for ns in batch.next_state if ns is not None])

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # compute Q(s_t, a) for the action taken
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        
        # compute V(s_{t+1}) for all the next states
        next_state_values = torch.zeros(self.batch_size)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values
        # compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # compute the loss
        loss = self.criterion(state_action_values, expected_state_action_values)

        # optimize the model
        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        return loss.detach()


    def train_epoch(self, arena :AvoidingArena):
        if arena.render_mode != "rgb_array":
            LOGGER.error("invalid <arena>'s render mode, must be 'rgb_array'")
            raise ValueError("invalid <arena>'s render mode, must be 'rgb_array'")

        global steps_done
        counter_succ, counter_fail, counter_opt, sum_loss = 0, 0, 0, 0.0
        iterator = tqdm(range(self.num_episodes))

        for i in iterator:
            _ = arena.reset()
            curr_state_img = self.arena.render()

            done = False
            while done:
                eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
                    math.exp(-1. * steps_done / self.eps_decay
                )
                steps_done += 1

                if torch.rand() < eps_threshold:
                    action = torch.tensor([self.arena.action_space.sample()])
                else:
                    action = self.predict_action(curr_state_img)
                
                _, reward, terminated, truncated, _ = self.arena.step(action[0])
                done = terminated or truncated