import os
import random
import pandas as pd

from operator import itemgetter
from typing import List, Tuple

from training import Trainer, METRICS
from states import *
from preprocessing import DATASET

from .config import REPLAY_MEMORY_CAPACITY, REWARD_FUNCTION
from .config import REPLAY_MEMORY_SAMPLE_SIZE

from .config import QLEARNING_LEARNING_RATE
from .config import QLEARNING_DISCOUNT_FACTOR
from .config import QLEARNING_EPSILON_STRATEGY


class SequenceGenerator:
    @staticmethod
    def _choose_action_type(state: State) -> str:
        prob = 0
        bins = []

        action_types = state.get_possible_action_types()
        action_probabilities = state.get_transition_probability()
        for action_type in action_types:
            prob += action_probabilities[action_type]
            bins.append(prob)

        assert len(bins) == len(action_types), "Error in bins length"
        assert round(bins[-1]) == 1, f"Sum of probabilities not equal 1.0 ({bins[-1]})"
        bins[-1] = 1

        random_number = random.random()

        for i in range(len(bins)):
            if random_number < bins[i]:
                return action_types[i]

    @staticmethod
    def choose_random_action(state: State) -> State:
        assert not isinstance(state, Output), "Cannot choose from Output state"

        action_type = SequenceGenerator._choose_action_type(state)
        actions = state.get_possible_actions()[action_type]
        action = random.choice(actions)
        return action

    @staticmethod
    def hash(sequence: List[State]) -> str:
        hash_str = '\n'.join([s.hash() for s in sequence])
        return hash_str

    @staticmethod
    def unhash(sequence_hashed: str) -> List[State]:
        sequence = [State.unhash(h) for h in sequence_hashed.split('\n')]
        return sequence


class QValues:
    def __init__(self, start_utilities=0.5):
        self.start_utilities = start_utilities
        self.q = self._create_q_table()
        QValues._create_q_table_folder()

    def _create_q_table(self):
        q_table = {}

        for state_type in State.get_all_state_types():
            for state in state_type.get_all_states():
                q_table[state.hash()] = {}
                possible_actions = state.get_possible_actions()
                for action_type in possible_actions:
                    for action in possible_actions[action_type]:
                        q_table[state.hash()][action.hash()] = self.start_utilities

        return q_table

    def save(self, epsilon):
        q_table = {
            'from_state': [],
            'to_state': [],
            'utility': []
        }

        for from_state in self.q:
            for to_state, utility in self.q[from_state].items():
                q_table['from_state'].append(from_state)
                q_table['to_state'].append(to_state)
                q_table['utility'].append(utility)

        df = pd.DataFrame(q_table)
        df.to_csv(os.path.join('logs', DATASET, 'q_table', f'eps_{epsilon:.1f}.csv'))

    @staticmethod
    def load(file_name):
        q_table = QValues()
        df = pd.read_csv(os.path.join('logs', DATASET, 'q_table', file_name))

        for _, row in df.iterrows():
            q_table.q[row['from_state']][row['to_state']] = row['utility']

        return q_table

    @staticmethod
    def _create_q_table_folder():
        if not os.path.exists(os.path.join('logs', DATASET, 'q_table')):
            os.mkdir(os.path.join('logs', DATASET, 'q_table'))


class ReplayMemory:
    def __init__(self):
        self.top = 0
        self.capacity = REPLAY_MEMORY_CAPACITY
        self.sample_size = REPLAY_MEMORY_SAMPLE_SIZE
        self.buffer: List[Tuple[str, dict, float]] = []
        ReplayMemory._create_memory_folder()

    def __contains__(self, sequence_hashed):
        return any([self.buffer[i][0] == sequence_hashed for i in range(len(self.buffer))])

    def add(self, sequence_hashed: str, evaluation: dict, epsilon: float):
        if self.top < self.capacity:
            self.buffer.append((sequence_hashed, evaluation, epsilon))
        else:
            self.buffer[self.top % self.capacity] = (sequence_hashed, evaluation, epsilon)
        self.top += 1

    def sample(self):
        return random.choices(self.buffer, k=self.sample_size)

    def save(self, epsilon):
        memory = {
            'sequence_hashed': [],
            'val_loss': [],
            'epsilon': []
        }
        for metric in METRICS:
            memory[f'val_{metric}'] = []

        for entry in self.buffer:
            memory['sequence_hashed'].append(entry[0])
            memory['val_loss'].append(entry[1]['val_loss'])
            memory['epsilon'].append(entry[2])
            for metric in METRICS:
                memory[f'val_{metric}'].append(entry[1][f'val_{metric}'])

        df = pd.DataFrame(memory)
        df.to_csv(os.path.join('logs', DATASET, 'memory', f'eps_{epsilon:.1f}.csv'))

    @staticmethod
    def load(file_name):
        memory = ReplayMemory()
        df = pd.read_csv(os.path.join('logs', DATASET, 'memory', file_name))

        for _, row in df.iterrows():
            evaluation = {'val_loss': row['val_loss']}
            for metric in METRICS:
                evaluation[f'val_{metric}'] = row[f'val_{metric}']

            memory.add(
                row['sequence_hashed'],
                evaluation,
                row['epsilon']
            )

        return memory

    @staticmethod
    def _create_memory_folder():
        if not os.path.exists(os.path.join('logs', DATASET, 'memory')):
            os.mkdir(os.path.join('logs', DATASET, 'memory'))


class QLearning:
    def __init__(self, data, q_values=None, memory=None):
        self.learning_rate = QLEARNING_LEARNING_RATE
        self.discount_factor = QLEARNING_DISCOUNT_FACTOR
        self.epsilon_strategy = QLEARNING_EPSILON_STRATEGY

        QLearning._create_log_folder()

        self.q_values = QValues() if q_values is None else q_values
        self.memory = ReplayMemory() if memory is None else memory

        self.data = data

    def learn(self):
        for epsilon, M in self.epsilon_strategy:
            episode = 1
            while episode <= M:
                print('Epsilon:', epsilon, 'Episode:', episode, '/', M)

                sequence = self.sample_network(epsilon)
                sequence_hashed = SequenceGenerator.hash(sequence)
                while sequence_hashed in self.memory:
                    sequence = self.sample_network(epsilon)
                    sequence_hashed = SequenceGenerator.hash(sequence)

                trainer = Trainer(sequence)
                history = trainer.train(self.data['X_train'], self.data['y_train'],
                                        self.data['X_val'], self.data['y_val'])
                evaluation = {'val_loss': history.history['val_loss'][-1]}
                for metric in METRICS:
                    evaluation[f'val_{metric}'] = history.history[f'val_{metric}'][-1]

                print('Network:', '--'.join(sequence_hashed.split('\n')))
                print('Evaluation:', evaluation)
                print()

                self.memory.add(sequence_hashed, evaluation, epsilon)

                for sample in self.memory.sample():
                    self._update_q_values_sequence(sample[0], REWARD_FUNCTION(sample[1]))

                episode += 1

            self.q_values.save(epsilon)
            self.memory.save(epsilon)

    def sample_network(self, epsilon) -> List[State]:
        state = Input()
        sequence = [state]

        while not isinstance(state, Output):
            alpha = random.random()
            if alpha < epsilon:  # Taking random action (next state)
                state = SequenceGenerator.choose_random_action(state)
            else:
                next_state_hash = max(self.q_values.q[state.hash()].items(), key=itemgetter(1))[0]
                state = State.unhash(next_state_hash)
            sequence.append(state)

        return sequence

    def _update_q_values_sequence(self, sequence_hashed, reward):
        sequence_hashed = sequence_hashed.split('\n')

        self._update_q_value(sequence_hashed[-2], sequence_hashed[-1], reward)

        for i in reversed(range(len(sequence_hashed) - 2)):
            self._update_q_value(sequence_hashed[i], sequence_hashed[i + 1], 0)

    def _update_q_value(self, start_state_hashed, to_state_hashed, reward):
        to_state = State.unhash(to_state_hashed)
        max_over_to_state = max(self.q_values.q[to_state_hashed].values()) if not isinstance(to_state, Output) else 0

        self.q_values.q[start_state_hashed][to_state_hashed] = \
            (1 - self.learning_rate) * self.q_values.q[start_state_hashed][to_state_hashed] + \
            self.learning_rate * (reward + self.discount_factor * max_over_to_state)

    @staticmethod
    def _create_log_folder():
        if not os.path.exists(os.path.join('logs', DATASET)):
            os.mkdir(os.path.join('logs', DATASET))
