import logging
import scallopy
import torch

from typing import *

from ..pacman.arena import AvoidingArena



ACTION = AvoidingArena.Actions
LOGGER = logging.getLogger(__name__)



class Agent():
    """
    """

    def __init__(self, grid_x :int, grid_y :int, eps :float=0.1, provenance :str='difftopkproofs'):
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.eps = eps
        self.provenance = provenance
        return


    def compute_next_action(self):
        #TODO: call neural component to "read" the board [NOT implemented yet]
        #TODO: call logical component to compute the next moves [implemented]
        pass


    def _logical_comp(self, agent_p :torch.Tensor, target_p :torch.Tensor, enemy_p :torch.Tensor) -> AvoidingArena.Actions:
        """Logical component of the model.

        Given a probabilistic classification of each cell in the game,
        compute the next move to be made to reach the target by the shortest path.

        Parameters
        ----------
        agent_p : torch.Tensor of shape (grid_x, grid_y)
            For each cell, probability that it contains the agent.
        target_p : torch.Tensor of shape (grid_x, grid_y)
            For each cell, probability that it contains the target.
        enemy_p : torch.Tensor of shape (grid_x, grid_y)
            For each cell, probability that it contains an enemy.

        Returns
        -------
        : AvoidingArena.Actions
            Next action to be performed.
        """
        nodes = [(i,j) for i in range(self.grid_x) for j in range(self.grid_y)]

        ctx = scallopy.ScallopContext(provenance=self.provenance)

        # create relations
        ctx.add_relation("actor", (int, int))
        ctx.add_relation("target", (int, int))
        ctx.add_relation("node", (int, int))

        # add facts to relations
        ctx.add_facts("actor", [(agent_p[node], node) for node in nodes])
        ctx.add_facts("target", [(target_p[node], node) for node in nodes])
        ctx.add_facts(
            "node",
            [(torch.clip(1 - enemy_p[node] - self.eps, min=0), node) for node in nodes],
        ) # <- self.eps> to penalise longer walks

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
            f" - agent -> {nodes[torch.argmax(agent_p)]} with prob. {torch.max(agent_p)}\n"
            f" - target -> {nodes[torch.argmax(target_p)]} with prob. {torch.max(target_p)}\n"
            f" - enemies -> {[node for node in nodes if enemy_p[node] > 0.5]} with prob. {enemy_p[enemy_p > 0.5].tolist()}"
        )
        LOGGER.info(f"next action: {tmp}")
        return actions[torch.argmax(probs)]


    def _neural_comp(self):
        pass