import tensorflow as tf


SUPPORTED_ENVS = {"Gridworld-v0", "Gridworld-v1", "Gridworld-v2", "4Rooms"}


# Basic model parameters.
tf.app.flags.DEFINE_string('game', 'MontezumaRevenge-v0',
                           """Bandit experiment type to be run""")
tf.app.flags.DEFINE_string('model_name', "FUN", """Name of the model""")
tf.app.flags.DEFINE_integer('resized_width', 84, """Resized width when using atari env""")
tf.app.flags.DEFINE_integer('resized_height', 84, """Resized height when using atari env""")
tf.app.flags.DEFINE_integer('agent_history_length', 3, """Agent history length""")
tf.app.flags.DEFINE_boolean('meta', True, "Whether to use meta")
tf.app.flags.DEFINE_boolean('resume', False,
                            """Resume training from latest checkpoint""")
tf.app.flags.DEFINE_boolean('train', True,
                            """Whether to train or test""")
tf.app.flags.DEFINE_boolean('show_training', False,
                            """Show windows with workers training""")
tf.app.flags.DEFINE_string('checkpoint_dir', './models',
                           """Directory where to save model checkpoints.""")
tf.app.flags.DEFINE_string('summaries_dir', './summaries',
                           """Directory where to write event logs""")
tf.app.flags.DEFINE_string('experiments_dir', './experiments',
                           """Directory where to write event experiments""")
tf.app.flags.DEFINE_string('test_experiments_dir', './test_experiments',
                           """Directory where to write event test experiments""")
tf.app.flags.DEFINE_string('frames_dir', './frames',
                           """Directory where to write event gifs of frames of model in case of FUN""")
tf.app.flags.DEFINE_boolean('monitor', False,
                            """Monitor test with gym monitor""")
tf.app.flags.DEFINE_integer('summary_interval', 1000, """Number of episodes of interval between summary saves""")
tf.app.flags.DEFINE_integer('checkpoint_interval', 1000, """Number of episodes of interval between checkpoint saves""")
tf.app.flags.DEFINE_integer('nb_actions', 4, """Number of actions to take""")
tf.app.flags.DEFINE_integer('nb_concurrent', 4, """Number of concurrent threads""")
tf.app.flags.DEFINE_integer('explore_steps', 900000, """Number of exploration steps""")
tf.app.flags.DEFINE_float('initial_random_goal_prob', 0.1, """Initial probability of exploration""")
tf.app.flags.DEFINE_float('final_random_goal_prob', 0, """final_random_goal_prob""")
tf.app.flags.DEFINE_float('w_gamma', 0.99, """Gamma value for worker""")
tf.app.flags.DEFINE_float('m_gamma', 0.99, """Gamma value for manager""")
tf.app.flags.DEFINE_float('lr', 1e-3, """Learning rate""")
tf.app.flags.DEFINE_float('w_beta_v', 0.25, """Coefficient of value function loss for worker""")
tf.app.flags.DEFINE_float('m_beta_v', 0.25, """Coefficient of value function loss for manager""")
tf.app.flags.DEFINE_float('beta_e', 1e-4, """Coefficient of entropy loss""")
tf.app.flags.DEFINE_integer('max_nb_episodes_train', 900000, """Max number of episodes of training time""")
tf.app.flags.DEFINE_float('gradient_clip_value', 50.0, """Gradient clip value for norm""")
tf.app.flags.DEFINE_integer('nb_test_episodes', 1, """Nb of test episodes""")
tf.app.flags.DEFINE_integer('BTT_length', 400, 'BTT length')
tf.app.flags.DEFINE_integer('hidden_dim', 256, 'The size of all the hidden layers')
tf.app.flags.DEFINE_integer('manager_horizon', 10, """The manager_horizon = r = c""")
tf.app.flags.DEFINE_integer('goal_embedding_size', 16, """The goal embedding size for the worker""")
tf.app.flags.DEFINE_integer('alpha', 0.5, """Alpha value to regulate the influence of the intrinsic reward
                            on the workers total reward""")
