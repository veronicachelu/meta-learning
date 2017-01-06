import tensorflow as tf

tf.app.flags.DEFINE_integer('SAVE_EVERY_X_STEPS', 600,
                            """Save interval.""")
tf.app.flags.DEFINE_float('FUTURE_REWARD_DISCOUNT', 0.99,
                          """Learning rate.""")
tf.app.flags.DEFINE_float('LEARN_RATE', 0.00001,
                          """Learning rate.""")
tf.app.flags.DEFINE_integer('PRINT_STATISTICS_EVERY_X_STEPS', 100,
                            """Print statistics interval.""")
tf.app.flags.DEFINE_integer('SUMMARY_INTERVAL', 5,
                            """Summary interval.""")
tf.app.flags.DEFINE_integer('EPISODES', 80000000,
                            """No of episodes.""")
tf.app.flags.DEFINE_integer('MINI_BATCH_SIZE', 1,
                            """size of mini batches.""")
tf.app.flags.DEFINE_integer('EXPLORE_STEPS', 1000000,
                            """Explore steps.""")
tf.app.flags.DEFINE_integer('STATE_FRAMES', 4,
                            """number of frames to store in the state.""")
tf.app.flags.DEFINE_float('INITIAL_RANDOM_ACTION_PROB', 1.0,
                          """starting chance of an action being random.""")
tf.app.flags.DEFINE_integer('RESIZED_SCREEN_X', 80,
                            """Screen size.""")
tf.app.flags.DEFINE_integer('RESIZED_SCREEN_Y', 80,
                            """Screen size.""")
tf.app.flags.DEFINE_integer('ENTROPY_BETA', 0.01,
                            """entropy param.""")
tf.app.flags.DEFINE_integer('STORE_SCORES_LEN', 200,
                            """How much memory the score vector should have.""")
tf.app.flags.DEFINE_integer('NUM_THREADS', 2,
                            """Number of threads.""")
tf.app.flags.DEFINE_integer('TARGET_NETWORK_UPDATE_FREQUENCY', 10000,
                            """Target network update frequency.""")
tf.app.flags.DEFINE_integer('NETWORK_UPDATE_FREQUENCY', 32,
                            """Network update frequency.""")
tf.app.flags.DEFINE_float('ENTROPY_BETA', 0.01, """Entropy param.""")
tf.app.flags.DEFINE_integer('MAX_TIME_STEPS', 32,
                            """Max no of time steps.""")
tf.app.flags.DEFINE_integer('batch_size', 2,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_boolean('SHOW_TRAINING', True,
                            """Wether to show the game window or not""")
tf.app.flags.DEFINE_string('CHECKPOINT_PATH', './checkpoints/checkpoint',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_string('SUMMARY_PATH', './summaries',
                           """Directory where to write event logs""")
tf.app.flags.DEFINE_string('EXPERIMENT', 'Breakout-v0',
                           """Name of the game""")




