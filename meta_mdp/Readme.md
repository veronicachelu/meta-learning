# Meta-Reinforcement Learning

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
    * ```Gridworld-x10-v0``` (10x10 gird world with 2 random colored squares same as above)
    
## Typical usage

### Training

*   To train a model you can run:

        $ python run.py
        
*   To evaluate a model you can run:

        $ python evaluate.py
        
*   To run a random agent baseline run:

        $ python run_baseline.py
        
*   To run an intelligent agent which knows the goal location and takes the minimum manhattan distance to the goal, run:
        
        $ python run_intelligent.py
    
## Tensorboard visualizations

* From ```summaries_dir``` run:
    
        $ tensorboard --logdir=.
    
* Watch the training visualizations at ```localhost:6006```

## Results

* Results are shown in this table, including with the best values for learning rates and gamma values:
        TODO

## Acknowledgements
- ```Learning to reinforcement learn - Jane X Wang, Zeb Kurth-Nelson, Dhruva Tirumala, Hubert Soyer, Joel Z Leibo, Remi Munos, Charles Blundell, Dharshan Kumaran, Matt Botvinick```
- [meta-rl](https://github.com/awjuliani/Meta-RL)

