"""
Alphazero learning agent
will be passed to HybridMCTS and PAMCTS_Outside
"""

# import modules
import numpy as np
from tensorflow import keras

from build_model import build_model


class alphazero_learning_agent:
    """
    Alphazero learning agent
    will be passed to HybridMCTS and PAMCTS_Outside
    """
    def __init__(self, weights_filepath):
        # self.model = keras.models.load_model(weights_filepath)
        self.model = build_model(num_hidden_layers=3)
        self.model.load_weights(weights_filepath)

    def get_q_value(self, child_state):
        """
        get q value of the current state
        """
        return self.model.predict(x=np.array(child_state).reshape((1, 4)), batch_size=1, verbose=0)[1][0][0]

    # def get_q_values(self, curr_state):
    #     """
    #     get q values of all children states
    #     """
    #     children_states = curr_state.tolist().chidren
    #     return np.array([self.get_q_value(state) for state in children_states])
    #
    # def get_greedy_action(self, curr_state):
    #     """
    #     get greedy action
    #     """
    #     children_q_values = self.get_q_values(curr_state)
    #     greedy_action = np.argmax(children_q_values)
    #     return greedy_action
