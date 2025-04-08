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

    def __ini__(
        self,
        arena :AvoidingArena,
        provenance :str='difftopkproofs',
        edge_penality :float=0.1
    ):
        """
        """
        self.arena = arena
        self.provenance = provenance
        self.edge_penality = edge_penality
        return


    def forward():
        #TODO: ...
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
            [(torch.clip(1 - enemy_p[node] - self.edge_penality, min=0), node) for node in self.nodes],
        ) # <- self.step_cost> to penalise longer walks

        # rules defining edges according to valid moves
        ctx.add_rule(f"edge(x, y, xp, y, {self.arena.Actions.RIGHT.value}) = node(x, y) and node(xp, y) and xp == x + 1")
        ctx.add_rule(f"edge(x, y, x, yp, {self.arena.Actions.UP.value})    = node(x, y) and node(x, yp) and yp == y + 1")
        ctx.add_rule(f"edge(x, y, xp, y, {self.arena.Actions.LEFT.value})  = node(x, y) and node(xp, y) and xp == x - 1")
        ctx.add_rule(f"edge(x, y, x, yp, {self.arena.Actions.DOWN.value})  = node(x, y) and node(x, yp) and yp == y - 1")
        
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
        actions = [self.arena.Actions(res[i][1][0]) for i in range(len(res))]
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
        #TODO: ...
        pass



class Agent():
    """
    """

    def __init__(
        self,
        arena :AvoidingArena,
        batch_size :int=32,
        memory_size :int=1024,
        gamma :float=0.99,
        eps_start :float=0.9,
        eps_end :float=0.05,
        eps_decay :float=1000,
        lr :float=1e-4,
        tau :float=5e-3,
        provenance :str='difftopkproofs'
    ):
        """
        Parameters
        ----------
        arena : AvoidingArena
            Arena of the game.
        batch_size : int, optional
            Batch size used to train the agent, by default ``32``.
        memory_size : int, optional
            Capacity of the agent's memory in number of transitions, by default ``1024``
        gamma : float, optional
            Discount factor of the Q-learning algorithm, by default ``0.99``.
        eps_start : float, optional
            Starting value of epsilon, by default ``0.9``.
            It determines the probability of choosing an action at random during training.
        eps_end : float, optional
            Final value of epsilon, by default ``0.05``.
        eps_decay : float, optional
            Controls the rate of exponential decay of epsilon, by default ``1000``.
        lr : float, optional
            learning rate of the optimizer, by default ``1e-4``.
        tau : float, optional
            update rate of the target network, by default ``5e-3``.
        provenance : str, optional
            Type of provenance used during execution, by default ``'difftopkproofs'``
        """
        self.arena = arena

        self.batch_size = batch_size
        self.memory_size = memory_size

        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.lr = lr
        self.tau = tau

        self.provenance = provenance

        self.training_steps_done = 0
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


    def select_action(self, rgb_array :torch.tensor) -> AvoidingArena.Actions:
        """
        """
        #TODO: controlla output type
        action_scores = self.policy_net(rgb_array)
        action = torch.argmax(action_scores, dim=1)
        return action


    def train(self, num_episodes :int, num_epochs :int) -> None:
        """
        Trains the agent using the specified number of episodes and epochs

        Parameters
        ----------
        num_episodes : int
            Number of episodes per epoch.
        num_epochs : int
            Number of epochs.
        """
        self.training_steps_done = 0
        for i in range(1, num_epochs + 1):
            self.train_epoch(num_episodes, i)
            self.test_epoch(num_episodes, i)
        return None


    def train_epoch(self, num_episodes :int, epoch :int=0) -> None:
        """
        Trains the agent for an epoch using the specified number of episodes.
        
        Parameters
        ----------
        num_episodes : int
            Number of episodes.
        epoch : int, optional
            Epoch considered, by default ``0``.

        Raises
        ------
        ValueError
            If the arena's render mode is not ``'rgb_arrary'``.
        """
        if self.arena.render_mode != "rgb_array":
            LOGGER.error("invalid <arena>'s render mode, must be 'rgb_array'")
            raise ValueError("invalid <arena>'s render mode, must be 'rgb_array'")

        iterator = tqdm(range(num_episodes))
        counter_succ, counter_upd, sum_loss = 0, 0, 0.0

        for i in iterator:
            _ = self.arena.reset()
            curr_state_image = torch.tensor(self.arena.render())

            done = False
            while not done:
                # select next action
                if random.rand() < self._eps_threshold():
                    action = torch.tensor([self.arena.action_space.sample()])
                else:
                    action = self.select_action(curr_state_image)
                
                _, reward, terminated, truncated, _ = self.arena.step(action)
                
                done = terminated or truncated
                next_state_image = None if done else self.arena.render()
                
                transition = Transition(curr_state_image, action, reward, done, next_state_image)
                self.memory.push(transition)
                
                # update policy network's weights
                loss = self._optimize()

                # soft update of the target network's weights θ′ ← τ θ + (1 −τ )θ′
                target_net_state_dict = self.target_net.state_dict()
                policy_net_state_dict = self.policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key]*self.tau + target_net_state_dict[key]*(1-self.tau)
                self.target_net.load_state_dict(target_net_state_dict)

                # update counters
                sum_loss += loss
                counter_upd += 1
                counter_succ += 1 if done and reward > 0 else 0

                # update current state
                curr_state_image = next_state_image

            # print training progress
            success_rate = (counter_succ / (i + 1)) * 100.0
            avg_loss = sum_loss / counter_upd
            iterator.set_description(f"[Train Epoch {epoch}] Avg Loss: {avg_loss}, Success: {counter_succ}/{i + 1} ({success_rate:.2f}%)")     
        return None


    def test_epoch(self, num_episodes :int, epoch :int=0) -> None:
        """
        Tests the agent for an epoch using the specified number of episodes.
        
        Parameters
        ----------
        num_episodes : int
            Number of episodes.
        epoch : int, optional
            Epoch considered, by default ``0``.

        Raises
        ------
        ValueError
            If the arena's render mode is not ``'rgb_arrary'``.
        """
        if self.arena.render_mode != "rgb_array":
            LOGGER.error("invalid <arena>'s render mode, must be 'rgb_array'")
            raise ValueError("invalid <arena>'s render mode, must be 'rgb_array'")
        
        iterator = tqdm(range(num_episodes))
        counter_succ = 0, 0

        for episode_i in iterator:
            _ = self.arena.reset()
            state_image = torch.tensor(self.arena.render())

            done = False
            while not done:
                # select next action
                action = self.select_action(state_image)
                _, reward, terminated, truncated, _ = self.arena.step(action)
            
                done = terminated or truncated
                state_image = torch.tensor(self.arena.render())

                # update counter
                counter_succ += 1 if done and reward > 0 else 0

            # print test progress
            success_rate = (counter_succ / (episode_i + 1)) * 100.0
            iterator.set_description(f"[Test Epoch {epoch}] Success {counter_succ}/{episode_i + 1} ({success_rate:.2f}%)")
        return None


    def _eps_threshold(self) -> float:
        """
        Updates epsilon according to the parameters provided 
        during the object's Creation and the number of training steps performed.
        
        Returns
        -------
        : float
            New value of epsilon.
        """
        eps = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * self.training_steps_done / self.eps_decay)
        self.training_steps_done += 1
        return eps


    def _optimize(self) -> float:
        """
        Update the agent's policy using a batch of transitions.
        
        Returns
        -------
        : float
            Current loss.
        
        Notes
        -----
        A number of transitions bigger than ``self.batch_size``
        must have taken place before the agent can actually be updated.
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