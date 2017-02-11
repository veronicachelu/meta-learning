import tensorflow as tf

# Basic model parameters.
tf.app.flags.DEFINE_string('game', 'Breakout-v0',
                           """experiment_name""")
tf.app.flags.DEFINE_string('GPU', "0",
                           """The GPU device to run on""")
tf.app.flags.DEFINE_boolean('resume', False,
                            """Resume training from latest checkpoint""")
tf.app.flags.DEFINE_boolean('show_training', True,
                            """""")
tf.app.flags.DEFINE_string('checkpoint_dir', './models/',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_string('summaries_dir', './summaries',
                           """Directory where to write event logs""")
tf.app.flags.DEFINE_string('frames_dir', './frames',
                           """Directory where to write event logs""")
tf.app.flags.DEFINE_integer('summary_interval', 5, """Start from epoch""")
tf.app.flags.DEFINE_integer('checkpoint_interval', 5000, """Start from epoch""")
# tf.app.flags.DEFINE_integer('nb_actions', 4, """Start from epoch""")
tf.app.flags.DEFINE_integer('nb_concurrent', 4, """Start from epoch""")
tf.app.flags.DEFINE_integer('nb_episodes', 20000, """Start from epoch""")
tf.app.flags.DEFINE_integer('max_nr_timesteps', 80000000, """Start from epoch""")
tf.app.flags.DEFINE_integer('max_episode_buffer_size', 32, """Start from epoch""")

tf.app.flags.DEFINE_integer('agent_history_length', 4, """Start from epoch""")
tf.app.flags.DEFINE_integer('resized_width', 84, """Start from epoch""")
tf.app.flags.DEFINE_integer('resized_height', 84, """Start from epoch""")
tf.app.flags.DEFINE_float('gamma', 0.99, """Learning rate""")
tf.app.flags.DEFINE_float('lr', 0.00001, """Learning rate""")
tf.app.flags.DEFINE_integer('initial_epoch', 0, """Start from epoch""")
