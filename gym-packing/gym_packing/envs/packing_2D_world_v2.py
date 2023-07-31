import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
from packutils.data.bin import Bin
from packutils.data.item import Item
from packutils.data.order import Order
from packutils.data.article import Article
from packutils.data.position import Position


class Packing2DWorldEnvV2(gym.Env):
    """
    Gym environment for 2D packing problems.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=(40, 20), use_height_map=True):
        """
        Initialize the packing environment.

        Args:
            render_mode (str): The rendering mode ("human" or "rgb_array").
            size (tuple): The size of the packing area (width, height).
        """
        self.size = size
        self.window_size = 512
        self.use_height_map = use_height_map

        # Create a sample order
        order = Order(
            order_id="test",
            articles=[
                Article(
                    article_id="article large",
                    width=4,
                    length=1,
                    height=4,
                    amount=10
                ),
                Article(
                    article_id="article small",
                    width=2,
                    length=1,
                    height=2,
                    amount=30
                )
            ])

        self._items = []
        for article in order.articles:
            for _ in range(article.amount):
                # Create item objects from the order articles
                self._items.append(Item.from_article(article))

        # Define the observation space
        grid_size = (self.size[0],) if self.use_height_map else self.size
        grid_max = self.size[1] + 1 if self.use_height_map else 1
        # print(grid_size)
        self.observation_space = spaces.Dict({
            # dimension and position of the item in 2D space
            "item": spaces.Box(low=0, high=max(size), shape=(4,), dtype=int),
            # 1 for allocated, 0 for free
            "grid": spaces.Box(low=0, high=grid_max, shape=grid_size, dtype=int),
        })

        # Define the action space
        self.action_space = spaces.Discrete(3)  # "right", "left", "pack"

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # Initialize PyGame window and clock for rendering
        self.window = None
        self.clock = None

    def _get_obs(self):
        """
        Get the current observation.

        Returns:
            dict: Dictionary containing the item location and the matrix representation of the grid.
        """
        item_obs = np.array([
            self._current_item.width,
            self._current_item.height,
            self._current_item.position.x,
            self._current_item.position.z
        ], dtype=int)
        return {"item": item_obs, "grid": self._matrix.flatten()}

    def _get_info(self):
        """
        Get additional information about the environment.

        Returns:
            dict: Additional information about the environment.
        """
        return {
            "packed": len(self._bin.packed_items),
            "remaining": len(self._items_to_pack)
        }

    @property
    def _matrix(self):
        """
        Get the matrix representation of the packing grid.

        Returns:
            numpy.ndarray: Matrix representation of the packing grid.
        """
        shape = self._bin.matrix.shape
        matrix = self._bin.matrix.reshape(shape[0], shape[2])

        if self.use_height_map:
            l = []
            for col in range(matrix.shape[1]):
                rows, = np.where(matrix[:, col] > 0)
                max_z = max(rows) + 1 if len(rows) > 0 else 0
                l.append(max_z)
            matrix = np.array(l, dtype=int)
        else:
            matrix[matrix > 0] = 1

        return matrix  # height x width

    def _move_item(self, step: 'int | None'):
        """
        Move the current item in the packing area.

        Args:
            step (int or None): The step size to move the item. If None, the item is reset to the leftmost position.
        """
        if step is None:
            new_x = self._bin.width // 2 - self._current_item.width // 2
        else:
            # Move left if step is -1, otherwise move right
            new_x = self._current_item.position.x + step
            if new_x < 0 or new_x + self._current_item.width > self._bin.width:
                return

        if self.use_height_map:
            z = max(self._matrix[new_x: new_x + self._current_item.width])
        else:
            rows, _ = np.where(
                self._matrix[:, new_x: new_x + self._current_item.width])
            z = max(rows) + 1 if len(rows) > 0 else 0
        self._current_item.position = Position(x=new_x, y=0, z=z)

    def reset(self, seed=None, options=None):
        """
        Reset the environment to its initial state.

        Args:
            seed (int): The random seed for the environment.
            options (dict): Additional options for resetting the environment.

        Returns:
            tuple: Initial observation and additional information.
        """
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Create a new packing bin
        self._bin = Bin(
            width=self.size[0],
            length=1,
            height=self.size[1]
        )

        self._items_to_pack = [item for item in self._items]
        self._current_item = self._items_to_pack[0]
        self._move_item(step=None)  # Reset position

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        """
        Take a step in the environment given an action.

        Args:
            action (int): The action to take.

        Returns:
            tuple: Next observation, reward, termination flag, additional information, and debug info.
        """
        done = False
        reward = 0

        if action == 0:  # Move left
            self._move_item(step=-1)
        elif action == 1:  # Move right
            self._move_item(step=1)
        elif action == 2:  # Pack the item
            is_packed, msg = self._bin.pack_item(self._current_item)
            if msg is not None:
                pass  # print(msg)
            if is_packed:
                reward += 10
                self._items_to_pack.remove(self._current_item)
                if len(self._items_to_pack) > 0:
                    self._current_item = self._items_to_pack[0]
                    self._move_item(step=None)
                else:
                    reward += 100
                    done = True
            else:
                reward -= 100
                done = True

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        # if done:
        #    print(observation, info, reward)
        return observation, reward, done, False, info

    def render(self):
        """
        Render the environment.

        Returns:
            numpy.ndarray: Rendered image if render_mode is "rgb_array", None otherwise.
        """
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        """
        Render a single frame of the environment.

        Returns:
            numpy.ndarray: Rendered image.
        """
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        # The size of a single grid square in pixels
        pix_square_size = self.window_size / max(self.size)

        for item in self._bin.packed_items:
            pygame.draw.rect(
                canvas,
                (255, 0, 0),
                pygame.Rect(
                    pix_square_size *
                    np.array([item.position.x, item.position.z], dtype=int),
                    (pix_square_size * item.width, pix_square_size * item.height),
                ),
            )

        position = np.array([self._current_item.position.x,
                            self._current_item.position.z], dtype=int)
        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (position + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # horizontal lines
        for x in range(self.size[1] + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (pix_square_size*self.size[0], pix_square_size * x),
                width=1,
            )
        # vertical lines
        for x in range(self.size[0] + 1):
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, pix_square_size * self.size[1]),
                width=1,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            self.window.blit(pygame.transform.flip(
                canvas, False, True), (0, 0))
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))

    def close(self):
        """
        Close the environment.
        """
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
