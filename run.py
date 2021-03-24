from learning import QLearning
from preprocessing import preprocess_data

data = preprocess_data()

learner = QLearning(data)
learner.learn()
