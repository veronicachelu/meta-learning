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
    * ```--frames_dir=```:  Directory where to save gifs of the training experiments. Default is ```./frames/```.
    * ```--frames_test_dir=```:  Directory where to save gifs of the testing experiments. Default is ```./frames_test/```.
    * ```--summary_interval=```:  Number of episodes of interval between summary saves. Default is ```5```.
    * ```--checkpoint_interval=```:  Number of episodes of interval between checkpoint saves. Default is ```250```.
    * ```--frames_interval=```:  Number of episodes of interval between training gifs saves. Default is ```100```.
    * ```--nb_concurrent=```:  Number of concurrent threads. Default is ```4```.
    * ```--gamma=```:  Gamma hyperparameter value. Default is ```0.8```.
    * ```--max_episode_buffer_size=```:  Buffer size between train updates. Default is ```32```.
    * ```--agent_history_length=```:  Number of frames that makes every state. Default is ```4```.
    * ```--resized_width=```:  Resized width of each frame. Default is ```84```.
    * ```--resized_height=```:  Resized height of each frame. Default is ```84```.
    * ```--lr=```:  Learning rate hyperparameter value. Default is ```1e-5```.
    * ```--beta_v=```:  Coefficient of value function loss. Default is ```0.05```.
    * ```--gradient_clip_value=```:  Clip the gradient of the weights if it overgrows this value. Default is ```40.0```.

## Tensorboard visualizations

* From ```summaries_dir``` run:
    
        $ tensorboard --logdir=.
    
* Watch the training visualizations at ```localhost:6006```
