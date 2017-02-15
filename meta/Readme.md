# Meta-Reinforcement Learning

## System requirements

 - Python3.5

## Python requirements

 - provided you are located in the root directory ```meta```, just run:

        $ sudo pip install -r requirements.txt

## Usage

    $ python run.py

#### Options

- can be modified by updating the ```flags.py``` file or by using the command line:

    * ```--game=```:  Env type: ```easy```, ```medium```, ```hard```, ```uniform```, ```independent```, ```11arms```, ```restless```. Default is ```easy```.
    * ```--resume=```:  Whether to use resume from a previous checkpoint. Default is False.
    * ```--train=```:  Whether to train or test. Default is ```True```.
    * ```--checkpoint_dir=```:  Directory where to save model checkpoints. Default is ```./models/```.
    * ```--summaries_dir=```:  Directory where to save model summaries in order to run tensorboard for visualizations. Default is ```./summaries/```.
    * ```--frames_dir=```:  Directory where to save gifs of the training experiments. Default is ```./frames/```.
    * ```--frames_test_dir=```:  Directory where to save gifs of the testing experiments. Default is ```./frames_test/```.
    * ```--summary_interval=```:  Number of episodes of interval between summary saves. Default is ```5```.
    * ```--checkpoint_interval=```:  Number of episodes of interval between checkpoint saves. Default is ```500```.
    * ```--frames_interval=```:  Number of episodes of interval between training gifs saves. Default is ```100```.
    * ```--nb_actions=```:  Number of actions to take. Default is ```2```.
    * ```--gamma=```:  Gamma hyperparameter value. Default is ```0.8```.
    * ```--lr=```:  Learning rate hyperparameter value. Default is ```1e-4```.
    * ```--beta_v=```:  Coefficient of value function loss. Default is ```0.05```.
    * ```--max_nb_episodes_train=```:  Max number of episodes of training time. Default is ```20000```.
    * ```--gradient_clip_value=```:  Clip the gradient of the weights if it overgrows this value. Default is ```50.0```.
    * ```--nb_test_episodes=```:  Number of testing environments to compute the mean regret and mean suboptimal number of arms on. Default is ```150```.


