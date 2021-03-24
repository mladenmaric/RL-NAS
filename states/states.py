from .config import *


class State:
    def __init__(self, state_type):
        self.state_type = state_type

    def __repr__(self):
        return self.state_type

    def get_possible_action_types(self):
        pass

    def get_possible_actions(self):
        pass

    def get_transition_probability(self):
        pass

    def hash(self) -> str:
        pass

    @staticmethod
    def unhash(hash_str: str):
        state_type = hash_str.split('|')[0]
        return eval(state_type).unhash(hash_str)

    @staticmethod
    def get_all_state_types():
        return [Input, Hidden, Dropout, Output]


class Input(State):
    def __init__(self):
        super().__init__(self.__class__.__name__)
        self.units = INPUT_LAYER_NEURONS
        self.layer_idx = 0

    def __repr__(self):
        return f'{super().__repr__()}(idx={self.layer_idx}, units={self.units})'

    # noinspection PyMethodMayBeStatic
    def get_possible_action_types(self):
        return ['Hidden', 'Output']

    def get_possible_actions(self):
        actions = {
            'Hidden': Hidden.get_states_by_layer(self.layer_idx + 1),
            'Output': Output.get_states_by_layer(self.layer_idx + 1)
        }

        return actions

    # noinspection PyMethodMayBeStatic
    def get_transition_probability(self):
        probability = {
            'Hidden': 1 - INPUT_OUTPUT_PROBABILITY,
            'Output': INPUT_OUTPUT_PROBABILITY
        }

        return probability

    def hash(self) -> str:
        return self.state_type

    @staticmethod
    def unhash(hash_str: str):
        return Input()

    @staticmethod
    def get_all_states():
        return [Input()]


class Hidden(State):
    def __init__(self, layer_idx, units, activation):
        super().__init__(self.__class__.__name__)
        self.layer_idx = layer_idx
        self.units = units
        self.activation = activation

    def __repr__(self):
        return f'{super().__repr__()}(idx={self.layer_idx}, units={self.units}, ' \
               f'activation={self.activation})'

    def get_possible_action_types(self):
        if self.layer_idx < HIDDEN_LAYER_MAX_DEPTH:
            return ['Hidden', 'Dropout', 'Output']
        else:
            return ['Dropout', 'Output']

    def get_possible_actions(self):
        actions = {}

        if self.layer_idx < HIDDEN_LAYER_MAX_DEPTH:
            actions['Hidden'] = Hidden.get_states_by_layer(self.layer_idx + 1)

        actions['Dropout'] = Dropout.get_states_by_layer(self.layer_idx)
        actions['Output'] = Output.get_states_by_layer(self.layer_idx + 1)

        return actions

    def get_transition_probability(self):
        output_probability = \
            HIDDEN_OUTPUT_PROBABILITY_START + \
            (HIDDEN_OUTPUT_PROBABILITY_END - HIDDEN_OUTPUT_PROBABILITY_START) / HIDDEN_LAYER_MAX_DEPTH * self.layer_idx
        remaining = 1 - output_probability

        probability = {'Output': output_probability}

        # Hidden : Dropout = 1 + units% : 1
        if self.layer_idx < HIDDEN_LAYER_MAX_DEPTH:
            k = remaining / (2 + self.units / HIDDEN_LAYER_MAX_UNITS)
            probability['Hidden'] = k * (1 + self.units / HIDDEN_LAYER_MAX_UNITS)
            probability['Dropout'] = k
        else:
            probability['Dropout'] = remaining

        return probability

    def hash(self) -> str:
        return '|'.join([self.state_type, str(self.layer_idx), str(self.units), self.activation])

    @staticmethod
    def unhash(hash_str: str):
        hash_str = hash_str.split('|')
        return Hidden(layer_idx=int(hash_str[1]), units=int(hash_str[2]), activation=hash_str[3])

    @staticmethod
    def get_all_states():
        states = []

        for layer_idx in range(1, HIDDEN_LAYER_MAX_DEPTH + 1):
            states.extend(Hidden.get_states_by_layer(layer_idx))

        return states

    @staticmethod
    def get_states_by_layer(layer_idx):
        states = []

        for units in range(HIDDEN_LAYER_MIN_UNITS, HIDDEN_LAYER_MAX_UNITS + 1, HIDDEN_LAYER_STEP_UNITS):
            for activation in HIDDEN_LAYER_ACTIVATIONS:
                states.append(Hidden(layer_idx, units, activation))

        return states


class Dropout(State):
    def __init__(self, layer_idx, rate):
        super().__init__(self.__class__.__name__)
        self.layer_idx = layer_idx
        self.rate = rate

    def __repr__(self):
        return f'{super().__repr__()}(idx={self.layer_idx}, rate={self.rate})'

    def get_possible_action_types(self):
        if self.layer_idx < HIDDEN_LAYER_MAX_DEPTH:
            return ['Hidden', 'Output']
        else:
            return ['Output']

    def get_possible_actions(self):
        actions = {}

        if self.layer_idx < HIDDEN_LAYER_MAX_DEPTH:
            actions['Hidden'] = Hidden.get_states_by_layer(self.layer_idx + 1)

        actions['Output'] = Output.get_states_by_layer(self.layer_idx + 1)

        return actions

    def get_transition_probability(self):
        probability = {}

        output_probability = \
            DROPOUT_OUTPUT_PROBABILITY_START + \
            (DROPOUT_OUTPUT_PROBABILITY_END - DROPOUT_OUTPUT_PROBABILITY_START) / HIDDEN_LAYER_MAX_DEPTH * \
            self.layer_idx

        if self.layer_idx < HIDDEN_LAYER_MAX_DEPTH:
            probability['Hidden'] = 1 - output_probability
            probability['Output'] = output_probability
        else:
            probability['Output'] = 1

        return probability

    def hash(self) -> str:
        return '|'.join([self.state_type, str(self.layer_idx), str(self.rate)])

    @staticmethod
    def unhash(hash_str: str):
        hash_str = hash_str.split('|')
        return Dropout(layer_idx=int(hash_str[1]), rate=float(hash_str[2]))

    @staticmethod
    def get_all_states():
        states = []

        for layer_idx in range(1, HIDDEN_LAYER_MAX_DEPTH + 1):
            states.extend(Dropout.get_states_by_layer(layer_idx))

        return states

    @staticmethod
    def get_states_by_layer(layer_idx):
        states = []

        for rate in range(DROPOUT_LAYER_MIN_RATE_PERCENT, DROPOUT_LAYER_MAX_RATE_PERCENT + 1,
                          DROPOUT_LAYER_STEP_RATE_PERCENT):
            rate /= 100.0
            states.append(Dropout(layer_idx, rate))

        return states


class Output(State):
    def __init__(self, layer_idx):
        super().__init__(self.__class__.__name__)
        self.units = OUTPUT_LAYER_NEURONS
        self.activation = OUTPUT_LAYER_ACTIVATION
        self.layer_idx = layer_idx

    def __repr__(self):
        return f'{super().__repr__()}(idx={self.layer_idx}, units={self.units}, ' \
               f'activation={self.activation})'

    def get_possible_action_types(self):
        return []

    def get_possible_actions(self):
        return {}

    def get_transition_probability(self):
        return {}

    def hash(self) -> str:
        return '|'.join([self.state_type, str(self.layer_idx)])

    @staticmethod
    def unhash(hash_str: str):
        hash_str = hash_str.split('|')
        return Output(layer_idx=int(hash_str[1]))

    @staticmethod
    def get_all_states():
        states = []

        for layer_idx in range(1, HIDDEN_LAYER_MAX_DEPTH + 2):
            states.extend(Output.get_states_by_layer(layer_idx))

        return states

    @staticmethod
    def get_states_by_layer(layer_idx):
        return [Output(layer_idx)]
