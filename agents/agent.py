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
from .utils import extract_cell, image_to_torch, Memory, Transition



LOGGER = logging.getLogger(__name__)



class CellClassifier(nn.Module):
    """
    """

    def __init__(self):
        super(CellClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=8, stride=4, padding=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=4, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=256, out_features=4),
            nn.Softmax(dim=1)
        )
        return


    def forward(self, x):
        return self.network(x)



class PolicyNet(nn.Module):
    """
    """

    def __init__(
        self,
        arena :AvoidingArena,
        provenance :str='difftopkproofs',
        edge_penality :float=0.1
    ):
        """
        """
        super(PolicyNet, self).__init__()

        self.arena = arena
        self.provenance = provenance
        self.edge_penality = edge_penality

        self.num_cells = arena.grid_x * arena.grid_y
        self.nodes = [(i,j) for i in range(arena.grid_x) for j in range(arena.grid_y)]

        self.cell_classifier = CellClassifier()

        self.path_planner = scallopy.Module(
            program = f"""
                type grid_node(x: usize, y: usize)

                // input from neural networks
                type agent(x: usize, y: usize)
                type target(x: usize, y: usize)
                type enemy(x: usize, y: usize)
                
                // safe nodes 
                rel node(x, y) = grid_node(x, y) and not enemy(x, y)
                
                // basic connectivity
                rel edge(x, y, xp, y, {self.arena.Actions.RIGHT.value}) = node(x, y) and node(xp, y) and xp == x + 1
                rel edge(x, y, x, yp, {self.arena.Actions.UP.value})    = node(x, y) and node(x, yp) and yp == y + 1
                rel edge(x, y, xp, y, {self.arena.Actions.LEFT.value})  = node(x, y) and node(xp, y) and xp == x - 1
                rel edge(x, y, x, yp, {self.arena.Actions.DOWN.value})  = node(x, y) and node(x, yp) and yp == y - 1
                
                // path for connectivity conditioned on no enemy on the path
                rel path(x, y, x, y) = node(x, y)
                rel path(x, y, xp, yp) = edge(x, y, xp, yp, _)
                rel path(x, y, xpp, ypp) = path(x, y, xp, yp) and edge(xp, yp, xpp, ypp, _)

                // get the next position
                rel next_position(xp, yp, a) = agent(x, y) and edge(x, y, xp, yp, a)
                rel next_action(a) = next_position(x, y, a) and path(x, y, gx, gy) and target(gx, gy)
            """,
            provenance = self.provenance,
            facts = {
                "node": [(torch.tensor(1 - self.edge_penality, requires_grad=False), node) for node in self.nodes]
            },
            input_mappings = {
                "agent": self.nodes,
                "target": self.nodes,
                "enemy": self.nodes
            },
            output_mappings = {
                "next_action": list(range(4))
            }
        )
        return


    def forward(self, x :torch.Tensor):
        """
        """
        batch_size, n_channel, *_ = x.shape

        if n_channel != 3 or x.ndim != 4:
            LOGGER.warning("<x>'s shape should be (B,C,H,W)")
        if not torch.is_floating_point(x):
            LOGGER.warning("<x>'s dtype should be a float")

        cells = torch.stack(
            [
                torch.stack(
                    [
                        extract_cell(x_i, node[0], node[1], self.arena.cell_size)
                        for node in self.nodes
                    ]
                )
                for x_i in x
            ]
        ).reshape(batch_size * self.num_cells, 3, self.arena.cell_size, self.arena.cell_size)

        features = self.cell_classifier(cells).reshape(batch_size, self.num_cells, 4)
        agent_p = features[:, :, 0]
        target_p =  features[:, :, 1]
        enemy_p = features[:, :, 2]

        next_actions = self.path_planner(agent=agent_p, target=target_p, enemy=enemy_p)
        next_action = torch.softmax(next_actions, dim=1)
        return next_action



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

        self.policy_net = PolicyNet(arena, provenance=provenance)
        self.target_net = PolicyNet(arena, provenance=provenance)
        
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.criterion = nn.SmoothL1Loss()
        self.optimizer = optim.RMSprop(self.policy_net.parameters(), self.lr)
        return


    def select_action(self, rgb_array :torch.Tensor|np.ndarray, preproc :bool=False) -> int:
        """
        """
        #TODO: controlla output type            
        if preproc:
            rgb_array = image_to_torch(rgb_array)
        if rgb_array.ndim == 3:
            rgb_array.unsqueeze_(0)

        action_scores = self.policy_net(rgb_array)
        action = torch.argmax(action_scores, dim=1)
        return int(action)


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

        iterator = tqdm.tqdm(range(num_episodes))
        counter_succ, counter_upd, sum_loss = 0, 0, 0.0

        for i in iterator:
            _ = self.arena.reset()
            curr_state_image = image_to_torch(self.arena.render())

            done = False
            while not done:
                # select next action
                if random.random() < self._eps_threshold():
                    action = self.arena.action_space.sample()
                else:
                    action = self.select_action(curr_state_image)
                
                _, reward, terminated, truncated, _ = self.arena.step(int(action))
                
                done = terminated or truncated
                next_state_image = None if done else image_to_torch(self.arena.render())
                
                transition = Transition(curr_state_image, action, reward, done, next_state_image)
                self.memory.push(transition)
                
                # update policy network's weights
                loss = self._optimize()

                # soft update of the target network's weights θ′ ← τ θ + (1 − τ) θ′
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
            state_image = Agent._img_to_torch(self.arena.render())

            done = False
            while not done:
                # select next action
                action = self.select_action(state_image)
                _, reward, terminated, truncated, _ = self.arena.step(action)
            
                done = terminated or truncated
                state_image = Agent._img_to_torch(self.arena.render())

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
            return 0.0
        
        # draw random transitions
        transitions = self.memory.sample(self.batch_size)
        # from batch-array of transitions to transition of batch-arrays
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(batch.done, dtype=torch.bool).logical_not()
        non_final_next_states = torch.stack([ns for ns in batch.next_state if ns is not None])

        state_batch = torch.stack(batch.state)
        action_batch = torch.tensor(batch.action)
        reward_batch = torch.tensor(batch.reward)

        # compute Q(s_t, a) for the action taken
        state_action_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1))
        
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