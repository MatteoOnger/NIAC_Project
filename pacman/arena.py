import gym
import logging
import numpy as np
import os

from gym import spaces
from typing import *



IMGS_DIR = os.path.join(os.path.dirname(__file__), "imgs")
LOGGER = logging.getLogger(__name__)



class AvoidingArena(gym.Env):
    """
    This class implements the arena of the game Pacman Maze
    by extending and respecting the standard defined by Gym.
    """

    metadata = {"render_modes": ["ansi", "human", "rgb_array"], "render_fps": 4}


    def __init__(
        self,
        render_mode :str|None=None,
        grid_dim :Tuple[int,int]=(5, 5),
        cell_size :int=64,
        max_num_moves :int=10,
        num_enemies :int=5,
        default_reward :float=0.0,
        on_success_reward :float=1.0,
        on_failure_reward :float=-1.0,
        remain_unchanged_reward :float=0.0,
    ):
        """
        Parameters
        ----------
        render_mode : str | None, optional
            Render mode to help visualise what the agent sees, by default ``None``.
        grid_dim : Tuple[int, int], optional
            A tuple of two integers for ``(grid_x, grid_y)``, by default ``(5, 5)``.
        cell_size : float, optional
            The side length of each cell in pixels, by default ``64``.
        max_num_moves : int, optional
            Maximum number of moves to solve the game, by default ``10``.
        num_enemies : int, optional
            Number of enemies, by default ``5``.
        default_reward : float, optional
            Default reward, by default ``0.0``.
        on_success_reward : float, optional
            Goal status reward, by default ``1.0``.
        on_failure_reward : float, optional
            Reward if hit by an enemy, by default ``-1.0``.
        remain_unchanged_reward : float, optional
            Reward for staying in the same position, by default ``0.0``.

        Raises
        ------
        ValueError
            - If the number of enemies is greater than the number of cells minus two
            (one cell for the start state and one for the goal state).
            - If the renderning mode is invalid.
        """
        if num_enemies > (grid_dim[0] * grid_dim[1] - 2):
            raise ValueError("Too many enemies")
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError("Render mode must be None, 'human' or 'rgb_array'")

        self.render_mode = render_mode
        self.grid_dim = np.array(grid_dim)
        self.cell_size = cell_size
        self.max_num_moves = max_num_moves
        self.num_enemies = num_enemies    
        self.default_reward = default_reward
        self.on_success_reward = on_success_reward
        self.on_failure_reward = on_failure_reward
        self.remain_unchanged_reward = remain_unchanged_reward

        self.grid_x, self.grid_y = grid_dim
        self.window_size_w, self.window_size_h = self.grid_x * self.cell_size, self.grid_y * self.cell_size

        # obeservations
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, self.grid_dim - 1, dtype=int),
                "goal": spaces.Box(0, self.grid_dim -1 , dtype=int),
                "enemies": spaces.Tuple(spaces.Box(0, 5 , dtype=int) for _ in range(3))
            }
        )

        # possible actions, corresponding to "right", "up", "left", "down"
        self.action_space = spaces.Discrete(4)
        self.action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
        }
        self.str_to_action = {
            "right": 0,
            "up": 1,
            "left": 2,
            "down": 3
        }

        # initialize environment states
        self.start_pos = None
        self.curr_pos = None
        self.goal_pos = None
        self.enemies = None
        self.moves_counter = None

        #  background agent, goal and enemy images
        self.background_image = None
        self.goal_image = None
        self.agent_image = None
        self.enemy_images = None 

        # if human-rendering is used, <self.window> will be a reference
        # to the window that we draw to. <self.clock> will be a clock that is used 
        # to ensure that the environment is rendered at the correct framerate 
        self.window_surface = None
        self.clock = None
        return


    def close(self) -> None:
        """
        After the user has finished using the environment, close contains the code necessary to "clean up" the environment.
        This is critical for closing rendering windows, database or HTTP connections.
        Calling ``close`` on an already closed environment has no effect and won't raise an error.
        """
        if self.window_surface is not None and self.render_mode in {"human", "rgb_array"}:
            try:
                import pygame
            except ImportError:
                raise ImportError("pygame is not installed") from None

            pygame.display.quit()
            pygame.quit()
            return None


    def render(self) -> np.ndarray|None:
        """
        Computes the render frames as specified by the attribute ``render_mode`` during the initialization of the environment.

        Returns
        -------
        : np.ndarray | None
            The rendering of the environment according to the specified mode.
        """
        if self.render_mode is None:
            LOGGER.warning("Calling render method without specifying any render mode")
        elif self.render_mode == "ansi":
            print(self)
        elif self.render_mode in {"human", "rgb_array"}:
            return self._render_frame()
        return None


    def reset(self, seed :int|None=None, options :Dict[str,Any]|None=None) -> Tuple[Dict[str,Any], Dict[str,Any]]:
        """
        Resets the environment to an initial internal state.

        Parameters
        ----------
        seed : int | None, optional
            The seed that is used to initialize the environment's PRNG (``np_random``) and
            the read-only attribute ``np_random_seed``.
            If the environment does not already have a PRNG and ``seed=None`` (the default option) is passed,
            a seed will be chosen from some source of entropy (e.g. timestamp or /dev/urandom).
            However, if the environment already has a PRNG and ``seed=None`` is passed, the PRNG will NOT be reset
            and the env's attribute ``np_random_seed`` will NOT be altered.
            If you pass an integer, the PRNG will be reset even if it already exists.
            Usually, you want to pass an integer right after the environment has been initialized and then never again.
        options : Dict[str, Any] | None, optional
            This parameter is ignored, present here for consistency, by default ``None``.

        Returns
        -------
        :Tuple[Dict[str, Any], Dict[str, Any]]
            Observation of the initial state and auxiliary infomration.
        """
        super().reset(seed=seed)
        empty_pos = [np.array([i,j]) for i in range(self.grid_x) for j in range(self.grid_y)]

        # init counter
        self.moves_counter = 0

        # generate start position
        idx = self.np_random.integers(0, len(empty_pos))
        self.start_pos = empty_pos[idx]
        del empty_pos[idx]

        # set current position
        self.curr_pos = self.start_pos

        # generate end position
        idx = self.np_random.integers(0, len(empty_pos))
        self.goal_pos = empty_pos[idx]
        del empty_pos[idx]

        # generate enemy positions
        idxs = self.np_random.choice(len(empty_pos), size=self.num_enemies, replace=False)
        self.enemies = np.array([empty_pos[idx] for idx in idxs])

        if self.render_mode == "human":
            self._render_frame()
        return (self._get_obs(), self._get_info())


    def step(self, action :int) -> Tuple[Dict[str,Any], float, bool, bool, Dict[str,Any]]:
        """
        Run one timestep of the environment's dynamics using the agent actions.
        When the end of an episode is reached (``terminated or truncated``), it is necessary to call the method ``reset`` to
        reset this environment's state for the next episode.

        Parameters
        ----------
        action : int
            An action provided by the agent to update the environment state.

        Returns
        -------
        observation : Dict[str,Any]
            An element of the environment's attribute ``observation_space`` as the next observation due to the agent actions.
        reward : float
            The reward as a result of taking the action.
        terminated : bool 
            Whether the agent reaches the terminal state (as defined under the MDP of the task)
            If true, the user needs to call the method ``reset``.
        truncated : bool
            Whether the truncation condition outside the scope of the MDP is satisfied, typically, this is a timelimit.
            Currently always set to ``False``.
        info : Dict[str,Any]
            Contains auxiliary diagnostic information (helpful for debugging, learning, and logging).
        """
        prev_pos = self.curr_pos
        direction = self.action_to_direction[action]

        # update agent position
        self.curr_pos = np.clip(
        self.curr_pos + direction, 0, self.grid_dim - 1
        )

        # update moves counter
        self.moves_counter += 1
        truncated = self.moves_counter >= self.max_num_moves

        # compute the reward
        reward, terminated = self.default_reward, False
        if (self.curr_pos == self.enemies).all(axis=1).any():
            reward, terminated = self.on_failure_reward, True
        elif (self.curr_pos == self.goal_pos).all():
            reward, terminated = self.on_success_reward, True
        elif (self.curr_pos == prev_pos).all():
            reward, terminated = self.remain_unchanged_reward, False

        if self.render_mode == "human":
            self._render_frame()
        return (self._get_obs(), reward, terminated, truncated, self._get_info())


    def _get_info(self) -> Dict[str,Any]:
        """
        Returns auxiliary infomration.

        Returns
        -------
        :Dict[str, Any]
            Auxiliary infomration.
        """
        return {"manhattan_distance": np.linalg.norm(self.curr_pos - self.goal_pos, ord=1)}


    def _get_obs(self) -> Dict[str,Any]:
        """
        Returns observation.

        Returns
        -------
        :Dict[str, Any]
            Observation.
        """
        return {"agent": self.curr_pos, "goal": self.goal_pos, "enemies": self.enemies}


    def _render_frame(self) -> np.ndarray|None:
        """
        Computes the render frames.

        Returns
        -------
        : np.ndarray | None
            The rendering of the environment according to the specified mode.

        Raises
        ------
        ImportError
            - If Pygame is not installed.
        """
        try:
            import pygame
        except ImportError:
            raise ImportError("Pygame is not installed") from None

        # init rendering
        if self.window_surface is None:
            pygame.init()

        if self.render_mode == "human":
            pygame.display.init()
            pygame.display.set_caption("Pacman Maze")
            self.window_surface = pygame.display.set_mode((self.window_size_w, self.window_size_h))
        elif self.render_mode == "rgb_array":
            self.window_surface = pygame.Surface((self.window_size_w, self.window_size_h))

        assert(self.window_surface is not None)

        if self.clock is None:
            self.clock = pygame.time.Clock()

        # load images
        if self.background_image is None:
            file_name = os.path.join(IMGS_DIR, "back.webp")
            self.background_image = pygame.transform.scale(
                pygame.image.load(file_name), (self.cell_size, self.cell_size)
            )
        if self.goal_image is None:
            file_name = os.path.join(IMGS_DIR, "flag.png")
            self.goal_image = pygame.transform.scale(
                pygame.image.load(file_name), (self.cell_size, self.cell_size)
            )
        if self.agent_image is None:
            file_name = os.path.join(IMGS_DIR, "agent.png")
            self.agent_image = pygame.transform.scale(
                pygame.image.load(file_name), (self.cell_size, self.cell_size)
            )
        if self.enemy_images is None:
            file_names = [
                os.path.join(IMGS_DIR, "enemy1.webp"),
                os.path.join(IMGS_DIR, "enemy2.webp")
                ]
            self.enemy_images = [
                pygame.transform.scale(
                    pygame.image.load(f_name), (self.cell_size, self.cell_size)
                ) for f_name in file_names
            ]

        for y in range(self.grid_y):
            for x in range(self.grid_x):
                pos = (x, y)
                img_pos = (x * self.cell_size, y * self.cell_size)
                rect = (*img_pos, self.cell_size, self.cell_size)

                self.window_surface.blit(self.background_image, img_pos)
                if (pos == self.curr_pos).all():
                    self.window_surface.blit(self.agent_image, img_pos)
                elif (pos == self.goal_pos).all():
                    self.window_surface.blit(self.goal_image, img_pos)
                elif (pos == self.enemies).all(axis=1).any():
                    self.window_surface.blit(self.enemy_images[(x+y)%2], img_pos)

                pygame.draw.rect(self.window_surface, (1, 50, 32), rect, 1)

        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.window_surface)), axes=(1, 0, 2)
            )
        return None


    def __str__(self) -> str:
        s = "┌" + ("─" * (2 * (self.grid_x + 2) - 3)) + "┐\n"
        for y in range(self.grid_y - 1, -1, -1):
            s += "│ "
            for x in range(self.grid_x):
                pos = np.array((x,y))
                if (pos == self.curr_pos).all():
                    s += "C "
                elif (pos == self.start_pos).all():
                    s += "S "
                elif (pos == self.goal_pos).all():
                    s += "G "
                elif (pos == self.enemies).all(axis=1).any():
                    s += "E "
                else:
                    s += "  "
            s += "│\n"
        s += "└" + ("─" * ((self.grid_x + 2) * 2 - 3)) + "┘\n"
        return s