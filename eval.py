import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from pprint import pprint
import random
import numpy as np
from training import Trainer, METRICS
from learning import QLearning, QValues, ReplayMemory, SequenceGenerator
from preprocessing import preprocess_data
from matplotlib import pyplot as plt

random.seed(None)

data = preprocess_data()
q_values = QValues.load(f'eps_0.1.csv')
memory = ReplayMemory.load(f'eps_0.1.csv')
learner = QLearning(data, q_values=q_values, memory=memory)

sampled_networks = []
best_networks = sorted(memory.buffer, key=lambda x: x[1]['val_r_square'], reverse=True)

for i in range(5):
    # network = learner.sample_network(epsilon=0.1)
    # while SequenceGenerator.hash(network) in sampled_networks:
    #     network = learner.sample_network(epsilon=0.1)
    # sampled_networks.append(SequenceGenerator.hash(network))

    network = best_networks[i][0]
    network = SequenceGenerator.unhash(network)

    pprint(network)
    trainer = Trainer(network)
    history = trainer.train(data['X_train'], data['y_train'], data['X_val'], data['y_val'])

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(f'{i + 1} Model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['training', 'val'], loc='upper left')
    plt.show()

    for metric in METRICS:
        plt.plot(history.history[f'{metric}'])
        plt.plot(history.history[f'val_{metric}'])
        plt.title(f'{i + 1} Metric {metric}')
        plt.ylabel(f'{metric}')
        plt.xlabel('epoch')
        plt.legend(['training', 'val'], loc='upper left')
        plt.show()

    y_test_pred = trainer.model.predict(data['X_test']).reshape(-1)
    plt.title(f'{i + 1} Test data correlation')
    plt.xlabel('Original value')
    plt.ylabel('Predicted value')
    plt.scatter(data['y_test'], y_test_pred)

    minimum = min(np.min(data['y_test']), np.min(y_test_pred))
    maximum = max(np.max(data['y_test']), np.max(y_test_pred))
    plt.plot([minimum, maximum], [minimum, maximum], color='r')
    plt.show()

    # X = np.concatenate((data['X_train'], data['X_val'], data['X_test']))
    # y = np.concatenate((data['y_train'], data['y_val'], data['y_test']))

    # X_training = np.concatenate((data['X_train'], data['X_val']))
    # y_training = np.concatenate((data['y_train'], data['y_val']))

    # y_test_pred = trainer.model.predict(data['X_test']).reshape(-1)
    # y_test_pred = np.where(y_test_pred > 0.5, 1, 0)

    print('Results on TRAIN set:', trainer.model.evaluate(data['X_train'], data['y_train'], verbose=0))
    print('Results on VAL set:', trainer.model.evaluate(data['X_val'], data['y_val'], verbose=0))
    print('Results on TEST set:', trainer.model.evaluate(data['X_test'], data['y_test'], verbose=0))
    # print('Error:', np.sum((y_test_pred - data['y_test']) ** 2) / len(y_test_pred))
    # print('Results on ALL sets:', trainer.model.evaluate(X, y))
