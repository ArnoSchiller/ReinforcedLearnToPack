import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data.dataset import Dataset, random_split

import gymnasium
from gymnasium.wrappers import FlattenObservation
from stable_baselines3 import PPO, A2C, SAC, TD3
from stable_baselines3.common.evaluation import evaluate_policy

import numpy as np
import random
import os

from packutils.data.article import Article
from packutils.data.order import Order
import constants as c

num_interactions = c.EXPERT_INTERACTIONS


def get_random_order():
    num_article_types = random.randint(1, 5)
    num_articles = random.randint(1, 20)

    articles = []
    for idx in range(num_article_types):
        dimensions = [random.randint(1, 40), random.randint(1, 40)]
        width, height = max(dimensions), min(dimensions)
        articles.append(Article(f"article {idx}", width, 1, height, amount=0))

    for _ in range(num_articles):
        index = random.randint(0, len(articles)-1)
        articles[index].amount += 1

    order = Order("test", articles=articles)
    return order


def generate_expert_dataset():

    data_path = f"expert_data_{c.MAX_NEXT_ITEMS}_rand_{num_interactions}.npz"
    if os.path.exists(data_path):
        with np.load(data_path) as data:
            expert_actions = data["expert_actions"]
            expert_observations = data["expert_observations"]
            return expert_actions, expert_observations

    env = gymnasium.make(
        c.ENVIRONMENT,
        articles=[],
        max_articles_per_order=None,
        max_next_items=c.MAX_NEXT_ITEMS,
        reward_strategies=c.REWARD_STRATEGIES,
        size=c.CONTAINER_SIZE,
        use_height_map=c.USE_HEIGHT_MAP,
        run_expert=True,
        # expert_image_dir=f"expert_images_{num_interactions}",
        render_mode=None)
    env = FlattenObservation(env)

    include_selection = c.MAX_NEXT_ITEMS > 0

    num_expert_actions = 2*num_interactions if include_selection else num_interactions
    expert_observations = np.zeros(
        (num_expert_actions,) + env.observation_space.shape, dtype=np.int32)
    expert_actions = np.zeros((num_expert_actions,), dtype=np.int64)
    actions = []
    dimensions = []
    for i in range(num_interactions):

        if len(actions) == 0:
            order = get_random_order()
            print(order)
            observation, _ = env.reset(options={"order": order})
            actions = env.expert_positions
            if include_selection:
                dimensions = env.expert_dimensions

        if include_selection:
            dimension = dimensions[0]
            dimensions.remove(dimension)
            matching_width = np.where(env.next_items[:, 0] == dimension[0])
            matching_height = np.where(env.next_items[:, 1] == dimension[1])
            intersect = np.intersect1d(matching_width, matching_height)
            action = env.size[0] + int(intersect)
            index = 2*i

            expert_actions[index] = int(action)
            expert_observations[index] = observation
            observation, reward, done, _, info = env.step(action)

        index = 2*i+1 if include_selection else i
        action = actions[0]
        actions.remove(action)
        expert_actions[index] = action
        expert_observations[index] = observation

        observation, reward, done, _, info = env.step(action)

    env.close()

    np.savez_compressed(
        data_path,
        expert_actions=expert_actions,
        expert_observations=expert_observations,
    )
    return expert_actions, expert_observations


class ExpertDataSet(Dataset):
    def __init__(self, expert_observations, expert_actions):
        self.observations = expert_observations
        self.actions = expert_actions

    def __getitem__(self, index):
        return (self.observations[index], self.actions[index])

    def __len__(self):
        return len(self.observations)


def pretrain_agent(
    student,
    env,
    batch_size=1,
    epochs=1000,
    scheduler_gamma=0.7,
    learning_rate=1.0,
    log_interval=100,
    no_cuda=True,
    seed=1,
    test_batch_size=1,
):
    use_cuda = not no_cuda and torch.cuda.is_available()
    torch.manual_seed(seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}
    criterion = nn.CrossEntropyLoss()
    # Extract initial policy
    model = student.policy.to(device)

    def train(model, device, train_loader, optimizer):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            dist = model.get_distribution(data)
            action_prediction = dist.distribution.logits
            target = target.long()
            loss = criterion(action_prediction, target)
            loss.backward()
            optimizer.step()

    def test(model, device, test_loader):
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                dist = model.get_distribution(data)
                action_prediction = dist.distribution.logits
                target = target.long()
                test_loss = criterion(action_prediction, target)
                test_loss /= len(test_loader.dataset)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_expert_dataset, batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_expert_dataset, batch_size=test_batch_size, shuffle=True, **kwargs,)

    optimizer = optim.Adadelta(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=1, gamma=scheduler_gamma)
    for epoch in range(1, epochs + 1):

        print(f"Running epoch {epoch} of {epochs}")
        train(model, device, train_loader, optimizer)
        test(model, device, test_loader)
        scheduler.step()

    student.policy = model


if __name__ == "__main__":
    expert_actions, expert_observations = generate_expert_dataset()

    expert_dataset = ExpertDataSet(expert_observations, expert_actions)
    train_size = int(0.8 * len(expert_dataset))
    test_size = len(expert_dataset) - train_size
    train_expert_dataset, test_expert_dataset = random_split(
        expert_dataset, [train_size, test_size])

    env = gymnasium.make(
        c.ENVIRONMENT,
        articles=c.ARTICLES,
        max_articles_per_order=c.MAX_ARTICLES,
        max_next_items=c.MAX_NEXT_ITEMS,
        reward_strategies=c.REWARD_STRATEGIES,
        size=c.CONTAINER_SIZE,
        use_height_map=c.USE_HEIGHT_MAP,
        run_expert=False,
        render_mode=None)
    env = FlattenObservation(env)

    import os
    TB_LOGS = os.path.join("training", "tb_logs_imitation")
    student = PPO(
        policy="MlpPolicy",
        env=env,
        ent_coef=0.01,
        # learning_rate=c.LEARNING_RATE,
        tensorboard_log=TB_LOGS
    )

    mean_reward, std_reward = evaluate_policy(
        student, env, n_eval_episodes=10)
    print(f"BEFORE TRAINING - Mean reward = {mean_reward} +/- {std_reward}")
    NUM_EPOCHS = 5_000
    pretrain_agent(
        student,
        env,
        epochs=NUM_EPOCHS,
        scheduler_gamma=0.7,
        learning_rate=1.0,
        log_interval=100,
        no_cuda=True,
        seed=1,
        batch_size=20,
        test_batch_size=1,)

    student.save(
        f"student_{c.MAX_NEXT_ITEMS}_rand_{num_interactions}_epochs_{NUM_EPOCHS}")
    mean_reward, std_reward = evaluate_policy(
        student, env, n_eval_episodes=10)
    print(f"AFTER TRAINING - Mean reward = {mean_reward} +/- {std_reward}")

""" 
BEFORE TRAINING - Mean reward = -650.0 +/- 0.0
"""
