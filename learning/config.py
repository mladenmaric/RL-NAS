REPLAY_MEMORY_CAPACITY = 256
REPLAY_MEMORY_SAMPLE_SIZE = 16

QLEARNING_LEARNING_RATE = 0.01
QLEARNING_DISCOUNT_FACTOR = 1.0
QLEARNING_EPSILON_STRATEGY = [
    (1.0, 100), (0.9, 30), (0.8, 30), (0.7, 30), (0.6, 30),
    (0.5, 30), (0.4, 30), (0.3, 30), (0.2, 30), (0.1, 30)
]


def REWARD_FUNCTION(evaluation):
    return evaluation['val_r_square']