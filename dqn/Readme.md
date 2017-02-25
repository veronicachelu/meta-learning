# DQN

## System requirements

 - Python3.5

## Python requirements

 - provided you are located in the root directory ```dqn```, just run:

        $ sudo pip install -r requirements.txt

## Usage

    $ python run.py

#### Options

- can be modified by updating the ```flags.py``` file or by using the command line:

    * ```--game=```:  Env name form openai gym. Default is ```Catcher-Level0-MetaLevel0-v0```.
    * ```--resume=```:  Whether to use resume from a previous checkpoint. Default is False.
    * ```--train=```:  Whether to train or test. Default is ```True```.
    * ```--show_training=```:  Show windows with workers training. Default is ```True```.
    * ```--checkpoint_dir=```:  Directory where to save model checkpoints. Default is ```./models/```.
    * ```--summaries_dir=```:  Directory where to save model summaries in order to run tensorboard for visualizations. Default is ```./summaries/```.
    * ```--experiments_dir=```:  Directory where to save experiments. Default is ```./experiments/```.
    * ```--summary_interval=```:  Number of episodes of interval between summary saves. Default is ```200```.
    * ```--test_performance_interval=```:  "Number of episodes of interval between testing reward performance. Default is ```1000```.
    * ```--checkpoint_interval=```:  Number of episodes of interval between checkpoint saves. Default is ```500```.
    * ```--nb_concurrent=```:  Number of concurrent threads. Default is ```4```.
    * ```--gamma=```:  Gamma hyperparameter value. Default is ```0.99```.
    * ```--agent_history_length=```:  Number of frames that makes every state. Default is ```4```.
    * ```--resized_width=```:  Resized width of each frame. Default is ```24```.
    * ```--resized_height=```:  Resized height of each frame. Default is ```24```.
    * ```--lr=```:  Learning rate hyperparameter value. Default is ```0.00025```.
    * ```--batch_size=```:  The size of the minibatch used for training. Default is ```128```.
    * ```--memory_size=```:  The size of the experience replay buffer. Default is ```1000000```.
    * ```--explore_steps=```:  The number of steps in which the probability of exploration is annealed . Default is ```1000000```.
    * ```--observation_steps=```:  The number of steps the agent just accumulates experience before starting training. Default is ```50000```.
    * ```--max_total_steps=```:  The total number of steps the agent will play in the environment. Default is ```5000000```.
    * ```--initial_random_action_prob=```:  The initial probability of exploration before annealing . Default is ```1.0```.
    * ```--final_random_action_prob=```:  The final probability of exploration after the exploration steps have ran out. Default is ```0.1```.
    * ```--TAO=```:  The rate at which the target network is updated from the q network. Default is ```0.001```.


## Example usage

    $ python run.py --game="Breakout-v0" --nb_concurrent=8 --resized_width=84 --resized_height=84 --max_episode_buffer_size=5 --conv1_nb_kernels=16 --conv2_nb_kernels=32 —conv1_kernel_size=8 --conv2_kernel_size=4 --conv1_stride=4 --conv2_stride=2 --conv1_padding='VALID' --conv2_padding='VALID' --fc_size=256 --lr=0.00025

## Tensorboard visualizations

* From ```summaries_dir``` run:
    
        $ tensorboard --logdir=.
    
* Watch the training visualizations at ```localhost:6006```
