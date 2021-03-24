import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from states import *
from training import Trainer, METRICS
from preprocessing import preprocess_data
import matplotlib.pyplot as plt
import numpy as np

data = preprocess_data()

network = [
    Input(),
    Hidden(1, 256, 'tanh'),
    Dropout(1, 0.3),
    Hidden(2, 128, 'relu'),
    Dropout(2, 0.2),
    Hidden(3, 64, 'relu'),
    Hidden(4, 32, 'relu'),
    Output(5)
]

trainer = Trainer(network)
history = trainer.train(data['X_train'], data['y_train'], data['X_val'], data['y_val'])

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['training', 'val'], loc='upper left')
plt.show()

for metric in METRICS:
    plt.plot(history.history[f'{metric}'])
    plt.plot(history.history[f'val_{metric}'])
    plt.title(f'Metric {metric}')
    plt.ylabel(f'{metric}')
    plt.xlabel('epoch')
    plt.legend(['training', 'val'], loc='upper left')
    plt.show()

plt.title('Test data')
plt.scatter(np.arange(len(data['y_test'])), data['y_test'])
y_test_pred = trainer.model.predict(data['X_test']).reshape(-1)
plt.scatter(np.arange(len(data['y_test'])), y_test_pred)
plt.show()

plt.title('Test data correlation')
plt.scatter(data['y_test'], y_test_pred)
plt.show()

# X = np.concatenate((data['X_train'], data['X_val'], data['X_test']))
# y = np.concatenate((data['y_train'], data['y_val'], data['y_test']))

X_training = np.concatenate((data['X_train'], data['X_val']))
y_training = np.concatenate((data['y_train'], data['y_val']))

# y_test_pred = trainer.model.predict(data['X_test']).reshape(-1)
# y_test_pred = np.where(y_test_pred > 0.5, 1, 0)

print('Results on TEST set:', trainer.model.evaluate(data['X_test'], data['y_test'], verbose=0))
# print('Error:', np.sum((y_test_pred - data['y_test']) ** 2) / len(y_test_pred))
print('Results on TRAIN i VAL sets:', trainer.model.evaluate(X_training, y_training, verbose=0))
# print('Results on ALL sets:', trainer.model.evaluate(X, y))
