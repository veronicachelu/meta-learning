# Meta-Reinforcement Learning with feudal networks for hierarchical reinforcement learning

## System requirements

 - Python3.5

## Python requirements

 - provided you are located in the root directory ```meta```, just run:

        $ sudo pip install -r requirements.txt

## Usage

    $ python run.py

#### Options

* can be modified by updating the ```flags.py``` file or by using the command line:

        $ python run.py -h
        
* multiple variants of the same environment are supported, i.e. the Gridworld env from:
        https://github.com/ioanachelu/gym_fast_envs

    * ```Gridworld-v0``` (5x5 world with 2 random colored squares. Color change every episode, but are constant between episodes.
    One of the colored squares is the target and gives reward 1. The other one gives reward 0. As soon as one square of a certain
    color is consumed a new one of the same color appears in a random location)
    * ```Gridworld-v2``` (5x5 world with 2 random colored squares. Fixed colors: the goal is the green one.
    One of the colored squares is the target and gives reward 1. The other one gives reward 0.
    * ```Gridworld-v1``` (5x5 world with 2 random colored squares. Simulated the non-matching environment from DeepMind maze
    in 2D. In the first room you see an object (colored square) and a teleporter(white square). The goal is to go to the 
    teleporter which gives a small reward of 1 and transports the agent to the next room where you have 2 colored squares. The goal
    is to collect the non-matching one (relative to the color of the object in the first room)
    * ```Gridworld-x10-v0``` (10x10 gird world with 2 random colored squares same as above)
    
## Typical usage

### Training

*   To train a model you can run:

        $ python run.py
        
*   To evaluate a model you can run:

        $ python evaluate.py

*   The evaluation procedure will also save in the frames directories the frames that maximize the goal relative to the crt state and 
the number of past states that maximize it.
        
## Tensorboard visualizations

* From ```summaries_dir``` run:
    
        $ tensorboard --logdir=.
    
* Watch the training visualizations at ```localhost:6006```

## Results

* Results for the NonMatching environment are show bellow:
![Alt text](./results/training_reward.jpg?raw=true "Training reward")
![Alt text](./results/training_length.jpg?raw=true "Training episode length")
![Alt text](./results/game.gif?raw=true "Trained agent")

## Acknowledgements

- ```FeUdal Networks for Hierarchical Reinforcement Learning - Alexander Sasha Vezhnevets, Simon Osindero, Tom Schaul, Nicolas Heess, Max Jaderberg, David Silver, Koray Kavukcuoglu```
- ```Learning to reinforcement learn - Jane X Wang, Zeb Kurth-Nelson, Dhruva Tirumala, Hubert Soyer, Joel Z Leibo, Remi Munos, Charles Blundell, Dharshan Kumaran, Matt Botvinick```
- [meta-rl](https://github.com/awjuliani/Meta-RL)

