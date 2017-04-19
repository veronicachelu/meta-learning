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
* multiple environments are supported:
    * ```independent``` (2 arms independent arms bandit)
    * ```uniform``` (2 arms with reward probabilities sampled from a uniform distribution [0,1])
    * ```easy``` (2 dependent arms with probabilities sampled from {0.1, 0.9})
    * ```medium``` (2 dependent arms with probabilities {0.25, 0.75})
    * ```hard``` (2 dependent arms with probabilites {0.4, 0.6})
    * ```11arms``` (11 arms bandit, one arm is the target arm with reward 5, arm 11 is always informative, i.e. reward 0.2 says that 
arm 2 is the target arm, reward 0.5 says that arm 5 is the target arm; the rest of the arms are suboptimal - give reward 1) - moved to a different folder

## Typical usage

* In order to run hyperparameter tuning with 100 model instances for learning rate and gamma value run:

        $ python hypertune.py
    
* Models are trained for 20 000 episodes, each episode consists of 100 time-steps
* Learning rate and gamma are sampled from:

    ```lr = 10 ** np.random.uniform(np.log10(10 ** (-2)), np.log10((10 ** (-4))))```
    
    ```gamma = np.random.uniform(0.7, 1.0)```
    
* Using the ```--resume=True``` flag resumes hyperparameter optimization

* Validating models on 150 sampled validation episodes each of them consisting of 100 time-steps:

        $ python validate_hypertune.py
    
* Results are saved to file: ```results_val.txt```

* In order to keeping only the best models, run:

        $ python evaluate_hypertune.py
    
* This saves to ```results_test.txt``` only the best 5 models. You can use one of them, i.e. use the learning rate and gamma value
to further evaluate models; If you have a saved model for easy with lr=0.002 gamma=0.85 you can use it like so to evaluate it
on a ```uniform``` environment for example

        $ python evaluate.py --game="uniform" --lr=0.002 --gamma=0.85
        
* You can also train a single model with your guessed choice of learning rate and gamma value using:

        $ python run.py --game="uniform" --lr=0.002 --gamma=0.85 
        
* In order to get the baseline value for a random model you can run:

        $ python run_baseline.py
         
* If something goes wrong while doing hyperparameter tunning and your ```models``` folder gets flooded with empty folders for models
that don't contain any checkpoints, you can run the following command which only keeps folders with trained models:

        $ python clean_models.py
        

## Tensorboard visualizations

* From ```summaries_dir``` run:
    
        $ tensorboard --logdir=.
    
* Watch the training visualizations at ```localhost:6006```

## Results

* Results are shown in this table, including with the best values for learning rates and gamma values:

