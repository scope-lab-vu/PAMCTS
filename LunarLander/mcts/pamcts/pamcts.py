from .mcts import MCTS
import time

class PA_MCTS:

    def __init__(self,
                gamma,
                learning_agent,
                alpha,
                num_iter,
                max_depth,
                c_puct,
                verbose=False):

        self.search_agent = MCTS(gamma=gamma,
                                learning_agent=learning_agent,
                                num_iter=num_iter,
                                max_depth=max_depth,
                                c_puct=c_puct)

        self.alpha = alpha
        self.learning_agent = learning_agent
        self.verbose = verbose

    def get_action(self, observation, env):

        '''
        1. get results from hybrid mcts - using get_action_scores and alpha = 0
        2. return action that arg maxes pa-uct equation
        :param observation:
        :param env:
        :return:
        '''
        # action_scores = self.search_agent.get_action_scores(env=env, observation=observation)
        # best_action = None
        # best_score = float('-inf')
        # for action_id, action_res in action_scores.items():
        #     score = self.get_pa_uct_score(policy_value=action_res['policy_value'],
        #                                   mcts_return=action_res['value'])
        #
        #     if self.verbose:
        #         print('\t{}: {} | {} | {}'.format(action_id, action_res['policy_value'], action_res['value'], score))
        #
        #     if score > best_score:
        #         best_action = action_id
        #         best_score = score
        #
        # if self.verbose:
        #     print('\tbest action: {} | {}'.format(best_action, best_score))
        #     print('---------')
        #
        # return best_action
        if self.alpha < 1.0:
            action_scores = self.search_agent.get_action_with_value(observation=observation, env=env)
        else:
            action_scores = None
        q_predict = self.learning_agent.get_q_values(observation)
        best_action = None
        best_score = float('-inf')

        for i in range(len(q_predict[0])):
            if self.alpha < 1.0:
                score = self.get_pa_uct_score(policy_value=q_predict[0][i],
                                        mcts_return=action_scores['value'][i])
            else:
                score = self.get_pa_uct_score(policy_value=q_predict[0][i],
                                        mcts_return=None)
            if score > best_score:
                best_action = i
                best_score = score

        return best_action

    def get_action_scores(self, observation, env):
        # raw_action_scores = self.search_agent.get_action_scores(env=env, observation=observation)
        # for action_id, action_res in raw_action_scores.items():
        #     score = self.get_pa_uct_score(policy_value=action_res['policy_value'],
        #                                   mcts_return=action_res['value'])
        #     action_res['en_score'] = score
        #
        # return raw_action_scores

        if self.alpha < 1.0:
            root_node = self.search_agent.get_action_with_value(observation=observation, env=env)
        else:
            root_node = None
        # ^ print(root_node)
        return root_node["value"]

    def get_pa_uct_score(self, policy_value, mcts_return):
        if self.alpha < 1.0:
            hybrid_node_value = (self.alpha * policy_value) + ((1.0 - self.alpha) * mcts_return)
        else:
            hybrid_node_value = self.alpha * policy_value
        #
        # # TODO | might want to include ucb1?
        return hybrid_node_value #+ self.C * np.sqrt(np.log(self.parent.visits)/self.visits)

    def clear_tree(self):
        """
        clear the MCTS
        """
        self.search_agent.clear_tree()
