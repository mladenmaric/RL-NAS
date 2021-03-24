from states import *
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, Nadam, RMSprop
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.callbacks import History
from tensorflow_addons.metrics import RSquare
from typing import List
from .config import *


class Trainer:
    def __init__(self, sequence: List[State]):
        self.sequence = sequence
        self.model = self._build_model()

    def _build_model(self) -> Sequential:
        model = Sequential()
        for state in self.sequence:
            if isinstance(state, Input):
                model.add(layers.Input(shape=(state.units,)))
            elif isinstance(state, Hidden):
                model.add(layers.Dense(units=state.units, activation=state.activation))
            elif isinstance(state, Dropout):
                model.add(layers.Dropout(rate=state.rate))
            elif isinstance(state, Output):
                model.add(layers.Dense(units=state.units, activation=state.activation))

        optimizer = OPTIMIZER
        if optimizer == 'adam':
            optimizer = Adam(LEARNING_RATE)
        elif optimizer == 'nadam':
            optimizer = Nadam(LEARNING_RATE)
        elif optimizer == 'rmsprop':
            optimizer = RMSprop(LEARNING_RATE)

        metrics = []
        for metric in METRICS:
            if metric == 'r_square':
                metric = RSquare(name=metric, y_shape=(1,))
            elif metric == 'rmse':
                metric = RootMeanSquaredError(metric)

            metrics.append(metric)

        model.compile(
            optimizer=optimizer,
            loss=LOSS,
            metrics=metrics
        )
        return model

    def train(self, X_train, y_train, X_val, y_val, epochs=EPOCHS, verbose=0) -> History:
        history = self.model.fit(
            x=X_train, y=y_train,
            batch_size=BATCH_SIZE,
            epochs=epochs,
            validation_data=(X_val, y_val),
            verbose=verbose
        )
        return history
