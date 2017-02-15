import tensorflow as tf

# Basic model parameters.
tf.app.flags.DEFINE_string('game', 'Breakout-v0',
                           """Experiment name from Atari platform""")
tf.app.flags.DEFINE_boolean('resume', False,
                            """Resume training from latest checkpoint""")
tf.app.flags.DEFINE_boolean('show_training', True,
                            """Show windows with workers training""")
tf.app.flags.DEFINE_string('checkpoint_dir', './models/',
                           """Directory where to save model checkpoints.""")
tf.app.flags.DEFINE_string('summaries_dir', './summaries',
                           """Directory where to write event logs""")
tf.app.flags.DEFINE_string('frames_dir', './frames',
                           """Directory where to write event gifs""")
tf.app.flags.DEFINE_string('frames_test_dir', './frames_test',
                           """Directory where to write test event gifs""")
tf.app.flags.DEFINE_integer('summary_interval', 5, """Number of episodes of interval between summary saves""")
tf.app.flags.DEFINE_integer('checkpoint_interval', 250, """Number of episodes of interval between checkpoint saves""")
tf.app.flags.DEFINE_integer('nb_concurrent', 4, """Number of concurrent threads""")
tf.app.flags.DEFINE_integer('max_episode_buffer_size', 32, """Buffer size between train updates""")
tf.app.flags.DEFINE_integer('agent_history_length', 4, """Number of frames that makes every state""")
tf.app.flags.DEFINE_integer('resized_width', 84, """Resized width of each frame""")
tf.app.flags.DEFINE_integer('resized_height', 84, """Resized height of each frame""")
tf.app.flags.DEFINE_float('gamma', 0.99, """Gamma value""")
tf.app.flags.DEFINE_float('lr', 1e-5, """Learning rate""")
tf.app.flags.DEFINE_float('beta_v', 0.5, """Coefficient of value function loss""")
tf.app.flags.DEFINE_float('beta_e', 0.01, """Coefficient of entropy function loss""")
tf.app.flags.DEFINE_float('gradient_clip_value', 40.0, """gradient_clip_value""")
