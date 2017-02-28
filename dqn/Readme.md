# DQN

## System requirements

 - Python3.5

## Python requirements

 - provided you are located in the root directory ```dqn```, just run:

        $ sudo pip install -r requirements.txt

## Usage

    $ python run.py

##### Options

A list of all available options is given by running:

    $python run.py -h

- can be modified by updating the ```flags.py``` file or by using the command line.


## Example usage

    $ python run.py --game="Breakout-v0" --nb_concurrent=8 --resized_width=84 --resized_height=84 --max_episode_buffer_size=5 --conv1_nb_kernels=16 --conv2_nb_kernels=32 —conv1_kernel_size=8 --conv2_kernel_size=4 --conv1_stride=4 --conv2_stride=2 --conv1_padding='VALID' --conv2_padding='VALID' --fc_size=256 --lr=0.00025

## Tensorboard visualizations

* From ```summaries_dir``` run:
    
        $ tensorboard --logdir=.
    
* Watch the training visualizations at ```localhost:6006```
