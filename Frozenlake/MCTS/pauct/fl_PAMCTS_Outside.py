from .stochasticMCTS import StochasticMCTS
import pandas as pd


class FL_PAMCTS_Outside():

    def __init__(self,
                 gamma,
                 learning_agent,
                 alpha,
                 num_iter,
                 max_depth,
                 c_puct,
                 verbose=False):

        self.search_agent = StochasticMCTS(gamma=gamma,
                                           learning_agent=None,
                                           alpha=0.0,
                                           num_iter=num_iter,
                                           max_depth=max_depth,
                                           c=c_puct
                                           )

        self.alpha = alpha
        self.learning_agent = learning_agent
        self.verbose=verbose

        pass

    def get_action(self, observation, env, prob_distribution):

        '''
        1. get results from hybrid mcts - using get_action_scores and alpha = 0
        2. return action that arg maxes pa-uct equation
        :param observation:
        :param env:
        :return:
        '''

        action_scores = self.search_agent.get_action_scores(env=env, observation=observation,
                                                            prob_distribution=prob_distribution)
        # print(action_scores) ## TODO | Ava debug
        best_action = None
        best_score = float('-inf')
        policy_values = self.learning_agent.get_q_values(observation)
        for action_id, action_res in action_scores.items():
            policy_value = policy_values[action_id]

            score = self.get_pa_uct_score(policy_value=policy_value,
                                          mcts_return=action_res['value'])
            # print(f"action_id: {action_id} | policy_value: {policy_value} | mcts_return: {action_res['value']} | score: {score}")

            # print('index: ', action_id, ' | q value: ', action_res['value'], '| uct_value: ', action_res['pamcts_value'], '| num_vists: ', action_res['num_visits'])

            if self.verbose:
                print('\t{}: {} | {} | {}'.format(action_id, policy_value, action_res['value'], score))
                print(f"# visited times: {action_res['num_visits']}")
                # write pro_disctribution, root state, action_id, policy_value, mcts_return, score, num_visits to csv files
                df = pd.DataFrame({'prob_distribution': [prob_distribution], 'root_state': [observation], 'action_id': [action_id], 'policy_value': [policy_value], 'mcts_return': [action_res['value']], 'score': [score], 'num_visits': [action_res['num_visits']]})
                df.to_csv('fl_high_gamma_simulations_test.csv', mode='a', header=False)

            if score > best_score:
                best_action = action_id
                best_score = score

        if self.verbose:
            print('\tbest action: {} | {}'.format(best_action, best_score))
            print('---------')

        return best_action

    def get_action_scores(self, observation, env, prob_distribution):

        raw_action_scores = self.search_agent.get_action_scores(env=env, observation=observation,
                                                                prob_distribution=prob_distribution)
        for action_id, action_res in raw_action_scores.items():
            score = self.get_pa_uct_score(policy_value=action_res['policy_value'],
                                          mcts_return=action_res['value'])
            action_res['en_score'] = score

        return raw_action_scores


    def get_pa_uct_score(self, policy_value, mcts_return):
        hybrid_node_value = (self.alpha * policy_value) + ((1.0 - self.alpha) * mcts_return)

        # TODO | might want to include ucb1?
        return hybrid_node_value #+ self.C * np.sqrt(np.log(self.parent.visits)/self.visits)




