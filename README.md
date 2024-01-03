# PAMCTS
## Decision Making in Non-Stationary Environments with Policy-Augmented Search

## Installation
```
pip install -r requirements.txt
```

Installation Error Fix: ModuleNotFoundError: No module named 'distutils.cmd'
```
sudo apt-get install python3.9-distutils
sudo apt-get install python3-apt
```

Box2d Installation Error Fix: error: command 'swig' failed: No such file or directory
```
sudo apt-get update
sudo apt-get -y install swig
```

Box2d Installation Error Fix: error: command '/usr/bin/x86_64-linux-gnu-gcc' failed with exit code 1
```
sudo apt-get update
sudo apt-get install python3.9-dev
sudo apt-get install g++
sudo apt-get install gcc
```

Box2d AttributeError: module '_Box2D' has no attribute 'RAND_LIMIT_swigconstant'
```
pip install box2d box2d-kengz
```

ERROR: google-auth 2.22.0 has requirement urllib3<2.0, but you'll have urllib3 2.0.4 which is incompatible.
```
pip install --upgrade botocore google-auth
```

## Run Experiments
### Cartpole:
Train DDQN Agent:
```
python Cartpole/train_masscart_dqn.py
```

Train Alphazero Network:
```
python Cartpole/Alphazero/training.py
```

Run PAMCTS:
```
python Cartpole/cartpole_pamcts.py
```

### FrozenLake:
Train DDQN Agent:
```
python Frozenlake/Network_Weights/DQN_3x3/frozenlake_reorder_dqn_retrain.py
```

Train Alphazero Network:
```
bash Frozenlake/Alphazero_Training/start.sh
```

Run PAMCTS:
```
python Frozenlake/flfl_pamcts.py
```

Run Alphazero:
```
python Frozenlake/flfl_alphazero.py
```

Run PAMCTS Alpha Selection:
```
python flfl_alpha_selection_part1.py
```

### Lunar Lander:
Train DDQN Agent:
```
python LunarLander/network_weights/DDQN/ddqn_agent_train.py
```

Train Alphazero Network:
```
python LunarLander/network_weights/Alphazero_Networks/training.py
```

Run PAMCTS:
```
python LunarLander/pamcts.py
```

Run Alphazero:
```
python LunarLander/alphazero.py
```

Run PAMCTS Alpha Selection:
```
python LunarLander/alpha_selection.py
```

### CliffWalking
Train DDQN Agent:
```
python CliffWalking/ddqn_agent.py
```

Train Alphazero Network:
```
python CliffWalking/alphazero_training.py
```

Run PAMCTS:
```
python CliffWalking/pamcts.py
```

Run Alphazero:
```
python CliffWalking/alphazero.py
```

Run PAMCTS Alpha Selection:
```
python CliffWalking/alpha_selection.py
```
