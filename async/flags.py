import tensorflow as tf

# Basic model parameters.
tf.app.flags.DEFINE_string('game', 'Catcher-Level0-MetaLevel0-v0',
                           """Experiment name from Atari platform""")
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
tf.app.flags.DEFINE_integer('summary_interval', 1000, """Number of episodes of interval between summary saves""")
tf.app.flags.DEFINE_integer('test_performance_interval', 1000,
                            """Number of episodes of interval between testing reward performance""")
tf.app.flags.DEFINE_integer('checkpoint_interval', 1000, """Number of episodes of interval between checkpoint saves""")
tf.app.flags.DEFINE_integer('nb_concurrent', 4, """Number of concurrent threads""")
tf.app.flags.DEFINE_integer('max_episode_buffer_size', 4, """Buffer size between train updates""")
tf.app.flags.DEFINE_integer('agent_history_length', 4, """Number of frames that makes every state""")
tf.app.flags.DEFINE_integer('resized_width', 24, """Resized width of each frame""")
tf.app.flags.DEFINE_integer('resized_height', 24, """Resized height of each frame""")
tf.app.flags.DEFINE_float('gamma', 0.99, """Gamma value""")
tf.app.flags.DEFINE_float('lr', 0.0007, """Learning rate""")
tf.app.flags.DEFINE_float('beta_v', 0.5, """Coefficient of value function loss""")
tf.app.flags.DEFINE_float('beta_e', 0.01, """Coefficient of entropy function loss""")
tf.app.flags.DEFINE_float('gradient_clip_value', 5.0, """gradient_clip_value""")
tf.app.flags.DEFINE_integer('seed', 1, """seed value for the gym env""")
tf.app.flags.DEFINE_integer('conv1_nb_kernels', 32, """conv1_nb_kernels""")
tf.app.flags.DEFINE_integer('conv2_nb_kernels', 32, """conv2_nb_kernels""")
tf.app.flags.DEFINE_integer('conv1_kernel_size', 5, """conv1_kernel_size""")
tf.app.flags.DEFINE_integer('conv2_kernel_size', 5, """conv2_kernel_size""")
tf.app.flags.DEFINE_integer('conv1_stride', 2, """conv1_stride""")
tf.app.flags.DEFINE_integer('conv2_stride', 2, """conv2_stride""")
tf.app.flags.DEFINE_string('conv1_padding', 'SAME', """conv1_padding""")
tf.app.flags.DEFINE_string('conv2_padding', 'VALID', """conv1_padding""")
tf.app.flags.DEFINE_integer('fc_size', 32, """fc_size""")
tf.app.flags.DEFINE_boolean('monitor', False,
                            """Monitor test with gym monitor""")
tf.app.flags.DEFINE_boolean('lstm', False,
                            """Whether to use lstm or not""")
tf.app.flags.DEFINE_boolean('gen_adv', False,
                            """Whether to use generalized advantage estimator or not""")
tf.flags.DEFINE_integer("eval_every", 300, "Evaluate the policy every N seconds")
tf.app.flags.DEFINE_boolean('meta', False,
                            """Whether to use meta-learning or not""")
tf.app.flags.DEFINE_boolean('verbose', False,
                            """Whether to display information about game dynamics""")
