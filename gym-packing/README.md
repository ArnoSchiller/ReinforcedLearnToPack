## TLDR
``` bash
# (optional)
python -m venv env
env\Scripts\activate

pip install -e .
pip install -r requirements.txt

# run random agent
python play_random.py

# train and run a model
python train_model.py
python play_model.py
```

## Concept
To build this project, the requirements are divided into different scenarios. To create a gymnasium environment, each scenario requires an `action_space`, an `observation_space`, and a reward strategy. With these definitions, a class inherited from the gymnasium environment can be created, which must override the `__init__`, `reset`, `step`, and `render` functions. 

### Scenario 1:
- 2D packing problem: a list of items are provided and the goal is to place as many items inside the bin as possible
-  `action_space`: 
   -  move_left
   -  move_right
   -  pack
- `observation_space`:
  - matrix of the bin representing the allocated and free space
  - current position of the item to pack
  - (list of remaining items to pack)
- `reward`:
  - for each correct placed item: +1 
  - for a wrong placement: -100 and game over
## ToDos
- [X] Include stability check when packing items (implemented in PackingUtils)
- [X] Redefine the action space as the possible solutions (Environment v2)
- [X] Train with another model (included DQN)

## Info / Source
This code is bases on the OpenAI Gymnasium [Tutorial](https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation) on building own environments. Also see [Gymnasium documentation](https://gymnasium.farama.org).
