import tensorflow as tf

# Basic model parameters.
tf.app.flags.DEFINE_string('game', 'Gridworld-v0',
                           """Bandit experiment type to be run""")
tf.app.flags.DEFINE_integer('game_size', 5, """Dimension of the gridworld""")
tf.app.flags.DEFINE_integer('game_channels', 3, """Nb of channels for each frame - rgb = 3""")
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
tf.app.flags.DEFINE_string('frames_dir', './frames',
                           """Directory where to write event gifs""")
tf.app.flags.DEFINE_boolean('monitor', False,
                            """Monitor test with gym monitor""")
tf.app.flags.DEFINE_boolean('meta', True,
                            """Whether to use meta learning framwork or not""")
tf.app.flags.DEFINE_integer('summary_interval', 500, """Number of episodes of interval between summary saves""")
tf.app.flags.DEFINE_integer('checkpoint_interval', 500, """Number of episodes of interval between checkpoint saves""")
tf.app.flags.DEFINE_integer('nb_actions', 4, """Number of actions to take""")
tf.app.flags.DEFINE_integer('nb_concurrent', 4, """Number of concurrent threads""")
tf.app.flags.DEFINE_float('gamma', 0.95, """Gamma value""")
tf.app.flags.DEFINE_float('lr', 1e-3, """Learning rate""")
tf.app.flags.DEFINE_float('beta_v', 0.05, """Coefficient of value function loss""")
tf.app.flags.DEFINE_integer('max_nb_episodes_train', 30000, """Max number of episodes of training time""")
tf.app.flags.DEFINE_float('gradient_clip_value', 50.0, """gradient_clip_value""")
tf.app.flags.DEFINE_integer('nb_test_episodes', 150, """Test episodes""")
tf.app.flags.DEFINE_boolean('gen_adv', True,
                            """Whether to use generalized advantage estimation""")
tf.app.flags.DEFINE_boolean('fw', True,
                            """Whether to use fast weights""")
