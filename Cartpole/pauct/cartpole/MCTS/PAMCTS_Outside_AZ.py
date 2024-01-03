from .HybridMCTS_AZ import HybridMCTS_AZ


class PAMCTS_Outside_AZ():

    def __init__(self,
                 gamma,
                 learning_agent,
                 alpha,
                 num_iter,
                 max_depth,
                 c_puct,
                 verbose=False):


        self.search_agent = HybridMCTS_AZ(gamma=gamma,
                                       learning_agent=learning_agent,
                                       alpha=0.0,
                                       num_iter=num_iter,
                                       max_depth=max_depth,
                                          c_puct=c_puct)

        self.alpha = alpha
        self.learning_agent = learning_agent
        self.verbose=verbose

        pass

    def get_action(self, observation, env):

        '''
        1. get results from hybrid mcts - using get_action_scores and alpha = 0
        2. return action that arg maxes pa-uct equation
        :param observation:
        :param env:
        :return:
        '''

        action_scores = self.search_agent.get_action_scores(env=env, observation=observation)
        best_action = None
        best_score = float('-inf')
        for action_id, action_res in action_scores.items():
            score = self.get_pa_uct_score(policy_value=action_res['policy_value'],
                                          mcts_return=action_res['value'])

            if self.verbose:
                print('\t{}: {} | {} | {}'.format(action_id, action_res['policy_value'], action_res['value'], score))

            if score > best_score:
                best_action = action_id
                best_score = score

        if self.verbose:
            print('\tbest action: {} | {}'.format(best_action, best_score))
            print('---------')

        return best_action

    def get_action_scores(self, observation, env):

        raw_action_scores = self.search_agent.get_action_scores(env=env, observation=observation)
        for action_id, action_res in raw_action_scores.items():
            score = self.get_pa_uct_score(policy_value=action_res['policy_value'],
                                          mcts_return=action_res['value'])
            action_res['en_score'] = score

        return raw_action_scores


    def get_pa_uct_score(self, policy_value, mcts_return):
        hybrid_node_value = (self.alpha * policy_value) + ((1.0 - self.alpha) * mcts_return)

        # TODO | might want to include ucb1?
        return hybrid_node_value #+ self.C * np.sqrt(np.log(self.parent.visits)/self.visits)




