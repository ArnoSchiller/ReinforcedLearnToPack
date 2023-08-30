import copy
import logging
import math
import os
import random
from typing import List
import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np

from packutils.data.bin import Bin
from packutils.data.item import Item
from packutils.data.order import Order
from packutils.data.position import Position
from packutils.data.article import Article
from packutils.data.packing_variant import PackingVariant
from packutils.visual.packing_visualization import PackingVisualization

from packutils.solver.palletier_packer import PalletierPacker

from gym_packing.envs.reward_strategies import RewardStrategy


class Packing2DWorldEnvV3(gym.Env):
    """
    Gym environment for 2D packing problems.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(
            self,
            articles,
            max_articles_per_order,
            max_next_items,
            reward_strategies: List[RewardStrategy],
            size=(40, 20),
            use_height_map=True,
            run_expert=False,
            expert_image_dir=None,
            render_mode=None,
    ):
        """
        Initialize the packing environment.

        Args:
            render_mode (str): The rendering mode ("human" or "rgb_array").
            size (tuple): The size of the packing area (width, height).
        """
        self.articles = articles
        self.max_articles_per_order = max_articles_per_order
        self._current_item = None
        self._items_to_pack = []

        self.size = size
        self.window_size = 512
        self.use_height_map = use_height_map
        self.run_expert = run_expert
        self.expert_idx = 0
        self.expert_image_dir = expert_image_dir
        self.max_next_items = max_next_items

        if self.run_expert and self.expert_image_dir is not None and os.path.exists(self.expert_image_dir):
            os.makedirs(self.expert_image_dir, exist_ok=True)

        if len(reward_strategies) < 1:
            raise ValueError("You must provide at least one reward strategy.")
        self.reward_strategies = reward_strategies

        # Define the observation space
        grid_size = (self.size[0],) if self.use_height_map else self.size
        grid_max = self.size[1] + 1 if self.use_height_map else 1
        # print(grid_size)

        obs_dict = {
            # dimension and position of the item in 2D space
            "item": spaces.Box(low=0, high=max(size), shape=(2,), dtype=int),
            # 1 for allocated, 0 for free or heightmap
            "grid": spaces.Box(low=0, high=grid_max, shape=grid_size, dtype=int),
        }
        if self.max_next_items > 0:
            obs_dict["next_items"] = spaces.Box(
                low=0, high=max(size), shape=(2*self.max_next_items,), dtype=int)

        self.observation_space = spaces.Dict(obs_dict)

        # Define the action space (each possible x position)
        num_actions = self.size[0] + self.max_next_items \
            if self.max_next_items > 0 else self.size[0]
        self.action_space = spaces.Discrete(num_actions)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # Initialize PyGame window and clock for rendering
        self.window = None
        self.clock = None

    def generate_order(self, order: 'Order | None' = None):
        if order is None:
            if self.max_articles_per_order is None:
                order = Order(
                    order_id="test",
                    articles=self.articles)
            else:
                _articles = copy.deepcopy(self.articles)
                for a in _articles:
                    a.amount = 0
                for _ in range(self.max_articles_per_order):
                    _articles[random.randint(0, len(_articles)-1)].amount += 1
                order = Order(
                    order_id="test",
                    articles=_articles)

        if self.run_expert:
            self.expert_idx += 1
            self.expert_dimensions = []
            self.expert_positions = []
            solver = PalletierPacker(bins=[self._bin])
            result = solver.pack_variant(order)

            if self.expert_image_dir is not None:
                try:
                    vis = PackingVisualization()
                    _ = vis.visualize_bin(
                        bin=result.bins[0], show=False, output_dir=self.expert_image_dir)
                except:
                    pass

            if result is None or len(result.bins) < 1:
                self.generate_order()
                return

            self._items = []
            for package in result.bins[0].packed_items:

                action = package.position.x
                self.expert_positions.append(action)
                dimension = [package.width, package.height]
                self.expert_dimensions.append(dimension)

                package.position = None
                self._items.append(package)

        else:
            logging.info(order)
            self._items = []
            for article in order.articles:
                for _ in range(article.amount):
                    # Create item objects from the order articles
                    self._items.append(Item.from_article(article))

    def _get_obs(self):
        """
        Get the current observation.

        Returns:
            dict: Dictionary containing the item location and the matrix representation of the grid.
        """
        item_obs = np.array([
            self._current_item.width,
            self._current_item.height
        ], dtype=int) if self._current_item is not None else np.array([0, 0], dtype=int)

        obs = {"item": item_obs, "grid": self._matrix}

        if self.max_next_items > 0:
            unique_dimensions = list(set([(i.width, i.height)
                                          for i in self._items_to_pack]))
            self.next_items = np.zeros((self.max_next_items, 2), dtype=int)
            for idx in range(min(len(unique_dimensions), self.max_next_items)):
                self.next_items[idx, :] = unique_dimensions[idx]
            obs["next_items"] = self.next_items.flatten()
        return obs

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
    def packed_variant(self) -> PackingVariant:
        """
        Get the packed bin as a packing variant.

        Returns:
            PackingVariant: Generated packing variant.
        """
        variant = PackingVariant()
        variant.add_bin(self._bin)
        variant.unpacked_items = []
        for unpacked in self._items_to_pack:
            variant.add_unpacked_item(unpacked, None)
        print(variant.unpacked_items)
        return variant  # height x width

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

    def reset(self, seed=None, options=None):
        """
        Reset the environment to its initial state.

        Args:
            seed (int): The random seed for the environment.
            options (dict): Additional options for resetting the environment.

        Returns:
            tuple: Initial observation and additional information.
        """
        order = None
        if options is not None:
            order = options.get("order", None)

        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Create a new packing bin
        self._bin = Bin(
            width=self.size[0],
            length=1,
            height=self.size[1]
        )

        if len(self.articles) > 0 or order is not None:
            self.generate_order(order=order)

            self._items_to_pack = [item for item in self._items]
            if len(self._items_to_pack) > 0 and not self.max_next_items > 0:
                self._current_item = self._items_to_pack[0]
                self._current_item.position = None
            else:
                self._current_item = None

        self.failed_counter = 0
        self.prev_max_z = 0
        self.prev_compactness = 0

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
        if self.max_next_items > 0:
            if action >= self.size[0]:
                # select an item
                index = action % self.size[0]
                selected_dim = self.next_items[index, :].tolist()

                possible_items = list(filter(
                    lambda x: x.width == int(
                        selected_dim[0]) and x.height == int(selected_dim[1]),
                    self._items_to_pack
                ))
                if len(possible_items) > 0:
                    self._current_item = possible_items[0]
                    return self._get_obs(), 0, False, False, {"info": f"item selected: {self._current_item}"}
                else:
                    return self._get_obs(), -50, False, False, {"info": f"no item to select {index}"}

            elif self._current_item is None:
                return self._get_obs(), -50, False, False, {"info": "no current item"}

        else:
            if self._current_item is None:
                return self._get_obs(), 0, True, False, {"error": "no current item"}

            if action >= self.size[0]:
                return self._get_obs(), 0, True, False, {"error": f"action not supported {action}"}

        done = False
        reward = 0

        self.prev_max_z = np.max(
            self._matrix, axis=0 if self.use_height_map else 1)
        self.prev_compactness = self.calculate_compactness()

        new_x = action
        if self.use_height_map:
            z = max(self._matrix[new_x: new_x + self._current_item.width])
        else:
            rows, _ = np.where(
                self._matrix[:, new_x: new_x + self._current_item.width])
            z = max(rows) + 1 if len(rows) > 0 else 0
        self._current_item.position = Position(x=int(new_x), y=0, z=int(z))

        is_packed, msg = self._bin.pack_item(self._current_item)

        if msg is not None:
            pass  # print(msg)

        if is_packed:
            self.failed_counter = 0
            self._items_to_pack.remove(self._current_item)
            if len(self._items_to_pack) < 1:
                done = True
        else:
            self.failed_counter += 1
            if self.failed_counter > 10:
                done = True

        reward = self.calculate_reward(is_packed, done)

        if not done and is_packed:
            if self.max_next_items > 0:
                self._current_item = None
            else:
                self._current_item = self._items_to_pack[0]

        if done and is_packed:
            self._current_item = None

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        # if done:
        #    print(observation, info, reward)
        return observation, reward, done, False, info

    def calculate_reward(self, is_packed, done):
        reward = 0

        # packed current item successfully
        if is_packed:
            if RewardStrategy.REWARD_EACH_PACKED in self.reward_strategies:
                reward += 20

            if RewardStrategy.REWARD_EACH_HEIGHT_CHANGE in self.reward_strategies:
                new_max_z = np.max(
                    self._matrix, axis=0 if self.use_height_map else 1)
                reward += 100 - (new_max_z - self.prev_max_z) * 10

            if RewardStrategy.REWARD_EACH_COMPACTNESS in self.reward_strategies:
                compactness = self.calculate_compactness()
                reward += 100 - (compactness - self.prev_compactness) * 10

            if RewardStrategy.PENALIZE_EACH_DISTANCE_COG in self.reward_strategies:
                cog = self._bin.get_center_of_gravity()
                pos = self._current_item.position
                distance_to_cog = int(math.sqrt(
                    (pos.x - cog.x)**2 + (pos.y - cog.y)**2 + (pos.z - cog.z)**2))
                reward -= distance_to_cog

            if RewardStrategy.PENALIZE_EACH_PACKED_HEIGHT in self.reward_strategies:
                reward -= self._current_item.position.z

            if RewardStrategy.PENALIZE_EACH_DISTANCE_NEXT_ITEM:
                x_min = self._current_item.position.x
                x_max = x_min + self._current_item.width
                z = self._current_item.position.z
                if self.use_height_map:
                    left_items = np.where(self._matrix[:x_min] > z)[0]
                    if len(left_items) > 0:
                        distance_left = x_min - np.max(left_items) - 1
                    else:
                        distance_left = x_min
                else:
                    raise NotImplementedError()

                reward -= 10 * distance_left if distance_left > 0 else -50

        # failed to pack current item
        else:
            if RewardStrategy.PENALIZE_EACH_PACKING_FAILED in self.reward_strategies:
                reward -= 50

        # last item packed successfully
        if done and is_packed:
            if RewardStrategy.REWARD_ALL_COMPACTNESS in self.reward_strategies:
                compactness = self.calculate_compactness()
                reward += compactness * 100

            if RewardStrategy.REWARD_ALL_PACKED in self.reward_strategies:
                reward += 100

        # failed to pack order
        if done and not is_packed:
            if RewardStrategy.PENALIZE_ALL_PACKING_FAILED in self.reward_strategies:
                reward -= 100
        return reward

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

    def calculate_compactness(self):
        if self.use_height_map:
            min_z = np.min(self._matrix, axis=0)
            max_z = np.max(self._matrix, axis=0)
            min_x = np.argmax(self._matrix > 0)
            max_x = self._matrix.shape[0] - \
                np.argmax(np.flip(self._matrix) > 0)
        else:
            max_z = np.max(self._matrix, axis=1)
            min_z = np.min(self._matrix, axis=1)

        used_volume = self._bin.get_used_volume()
        allocated_volume = (max_z - min_z) * (max_x - min_x)

        if allocated_volume == 0:
            return 1

        compactness = used_volume / allocated_volume
        return compactness
