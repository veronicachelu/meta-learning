# A3C

## System requirements

 - Python3.5

## Python requirements

 - provided you are located in the root directory ```async```, just run:

        $ sudo pip install -r requirements.txt

## Usage

    $ python run.py

#### Options

- can be modified by updating the ```flags.py``` file or by using the command line:

    * ```--game=```:  Env name form openai gym. Default is ```Catcher-v0```.
    * ```--resume=```:  Whether to use resume from a previous checkpoint. Default is False.
    * ```--train=```:  Whether to train or test. Default is ```True```.
    * ```--show_training=```:  Show windows with workers training. Default is ```True```.
    * ```--checkpoint_dir=```:  Directory where to save model checkpoints. Default is ```./models/```.
    * ```--summaries_dir=```:  Directory where to save model summaries in order to run tensorboard for visualizations. Default is ```./summaries/```.
    * ```--experiments_dir=```:  Directory where to save experiments. Default is ```./experiments/```.
    * ```--summary_interval=```:  Number of episodes of interval between summary saves. Default is ```5```.
    * ```--test_performance_interval=```:  "Number of episodes of interval between testing reward performance. Default is ```1000```.
    * ```--checkpoint_interval=```:  Number of episodes of interval between checkpoint saves. Default is ```20```.
    * ```--nb_concurrent=```:  Number of concurrent threads. Default is ```4```.
    * ```--gamma=```:  Gamma hyperparameter value. Default is ```0.99```.
    * ```--max_episode_buffer_size=```:  Buffer size between train updates. Default is ```4```.
    * ```--agent_history_length=```:  Number of frames that makes every state. Default is ```4```.
    * ```--resized_width=```:  Resized width of each frame. Default is ```24```.
    * ```--resized_height=```:  Resized height of each frame. Default is ```24```.
    * ```--lr=```:  Learning rate hyperparameter value. Default is ```0.00025```.
    * ```--beta_v=```:  Coefficient of value function loss. Default is ```0.5```.
    * ```--beta_e=```:  Coefficient of entropy function loss. Default is ```0.01```.
    * ```--gradient_clip_value=```:  Clip the gradient of the weights if it overgrows this value. Default is ```40.0```.
    * ```--monitor=```:  Monitor test with gym monitor. Default is ```False```.

## Example usage


    $ python run.py --game="Breakout-v0" --nb_concurrent=8 --resized_width=84 --resized_height=84 --max_episode_buffer_size=5
    --conv1_nb_kernels=16 --conv2_nb_kernels=32 --conv1_kernel_size=8 --conv2_kernel_size=4 --conv1_stride=4 --conv2_stride=2
    --conv1_padding='VALID' --conv2_padding='VALID' --fc_size=256 --lr=0.00025


## Tensorboard visualizations

* From ```summaries_dir``` run:
    
        $ tensorboard --logdir=.
    
* Watch the training visualizations at ```localhost:6006```
