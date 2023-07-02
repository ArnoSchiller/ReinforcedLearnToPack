import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np

from packutils.data.bin import Bin
from packutils.data.item import Item
from packutils.data.order import Order
from packutils.data.article import Article
from packutils.data.position import Position


class Packing2DWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=(5, 5)):
        self.size = size
        self.window_size = 512  # The size of the PyGame window

        order = Order(
            order_id="test",
            articles=[
                Article(
                    article_id="article 1",
                    width=2,
                    length=1,
                    height=2,
                    amount=5
                )
            ])

        self._items = []
        for article in order.articles:
            for _ in range(article.amount):
                # replace with method Item.from_article(article)
                self._items.append(Item(
                    id=article.article_id,
                    width=article.width,
                    length=article.length,
                    height=article.height,
                    weight=article.weight
                ))

        # Observations are dictionaries with the item location to pack and the matrix representation of the grid.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict(
            {
                # dimension and position of the item in 2D space
                "item": spaces.Box(low=0, high=max(size) - 1, shape=(4,), dtype=int),
                # 1 for allocated, 0 for free
                "grid": spaces.Box(low=0, high=1, shape=size, dtype=int),
            }
        )

        # We have 3 actions, corresponding to "right", "left", "pack"
        self.action_space = spaces.Discrete(3)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    def _get_obs(self):
        item_obs = np.array([
            self._current_item.width,
            self._current_item.height,
            self._current_item.position.x,
            self._current_item.position.z
        ], dtype=int)
        return {"item": item_obs, "grid": self._matrix}

    def _get_info(self):
        return {
            "test": 0,
            "bin": self._bin.packed_items
        }

    @property
    def _matrix(self):
        shape = self._bin.matrix.shape
        matrix = self._bin.matrix.reshape(shape[0], shape[2])
        return matrix  # height x width

    def _move_item(self, step: 'int | None'):
        if step is None:
            new_x = 0
        else:
            # moves left if step is -1 else right
            new_x = self._current_item.position.x + step
            print("new_x", new_x)
            if new_x < 0 or new_x + self._current_item.width > self._bin.width:
                return
        rows, _ = np.where(
            self._matrix[:, new_x: new_x+self._current_item.width])
        z = max(rows) + 1 if len(rows) > 0 else 0
        self._current_item.position = Position(
            x=new_x, y=0, z=z
        )
        print(self._current_item.position.x, self._current_item.position.z)

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        self._bin = Bin(
            width=self.size[0],
            length=1,
            height=self.size[1]
        )

        self._items_to_pack = self._items
        self._current_item = self._items_to_pack[0]
        self._move_item(step=None)  # reset position

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        failed_to_pack = False
        print(f"action: {action}")
        if action == 0:  # move left
            self._move_item(step=-1)
        elif action == 1:  # move right
            self._move_item(step=1)
        elif action == 2:
            # pack the item and update remaining items
            is_packed, msg = self._bin.pack_item(self._current_item)
            if msg is not None:
                print(msg)
            if is_packed:
                self._items_to_pack.remove(self._current_item)
                if len(self._items_to_pack) > 0:
                    self._current_item = self._items_to_pack[0]
                    self._move_item(step=None)
            else:
                failed_to_pack = True
            print(self._current_item.position.x, self._current_item.position.z)
        # An episode is done if no items left
        terminated = (len(self._items_to_pack) == 0) or failed_to_pack
        reward = 1 if terminated else 0  # Binary sparse rewards
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / max(self.size)
        )  # The size of a single grid square in pixels

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

        # Finally, add some gridlines
        for x in range(max(self.size) + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
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
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
