from .stochasticMCTSNode import ActionNode, StateNode
import copy
from gym import spaces
import itertools
from .discrete_w_start import DiscreteWS
from random import choice
from .customized_frozen_lake import Customized_FrozenLakeEnv

custom_map = [
    "SHF",
    "FFF",
    "HFG"
]

class Stochastic_graph_search:

    def __init__(self,
                 gamma = 0.9,
                 learning_agent=None,
                 alpha=0.0,
                 num_iter=10000,
                 max_depth=500,
                 c=500.0):  # c = 1.4142

        self.num_iterations = num_iter
        self.max_depth = max_depth
        self.learning_agent = learning_agent
        self.gamma = gamma
        self.alpha = alpha
        self.c = c
        self.root = StateNode(parent_action_node=None,
                              state=None, # set this at some point
                              depth=0,
                              c=self.c)

    def combinations(self, space):
        if isinstance(space, DiscreteWS):
            return range(space.start, space.start + space.n) # support kserver
        if isinstance(space, spaces.Discrete):
            return range(space.n)
        elif isinstance(space, spaces.Tuple):
            return itertools.product(*[self.combinations(s) for s in space.spaces])
        else:
            raise NotImplementedError


    def create_state_node(self,
                          action_node,
                          new_state,
                          depth,
                          search_env,
                          done):

        curr_state_node = StateNode(parent_action_node=action_node,
                                    state=new_state,
                                    depth=depth,
                                    c=self.c)
        if action_node is not None:
            action_node.possible_child_state_nodes.append(curr_state_node)
        if done is not None:
            if self.learning_agent is not None:
                policy_values = self.learning_agent.get_q_values(new_state)
                curr_state_node.child_action_nodes = [ActionNode(parent_state_node=curr_state_node,
                                              action=a,
                                              policy_value=policy_values[a],
                                              alpha=self.alpha) for a in self.combinations(search_env.action_space)]

            else:
                curr_state_node.child_action_nodes = [ActionNode(parent_state_node=curr_state_node,
                                                                 action=a,
                                                                 policy_value=None,
                                                                 alpha=self.alpha) for a in self.combinations(search_env.action_space)]

        return curr_state_node


    def compute_plan(self, env, starting_state, prob_distribution):
        """
        Assume starting state is not terminal
        :param env:
        :param starting_state:
        :return:
        """
        search_env = Customized_FrozenLakeEnv(desc=custom_map, is_slippery=True, customized_prob1=prob_distribution[0],
                                              customized_prob2=prob_distribution[1],
                                              customized_prob3=prob_distribution[2])
        # search_env = copy.deepcopy(env)
        search_env._max_episode_steps = 200
        search_env.s = starting_state
        # search_env = copy.deepcopy(env)
        done = False
        self.root = self.create_state_node(action_node=None,
                                           new_state=copy.deepcopy(search_env.s),
                                           depth=0,
                                           search_env=search_env,
                                           done=done)



        for iteration in range(self.num_iterations):

            search_env.reset()
            search_env.s = self.root.state

            curr_state_node = self.root
            curr_state = curr_state_node.state
            curr_depth = 0
            done = False
            trajectory_rewards = []
            # action_node = max(curr_state_node.child_action_nodes,
            #                   key=lambda node: node.ucb_score)


            while True: #curr_state_node.child_action_nodes:
                mx_ucb = max([_.ucb_score for _ in curr_state_node.child_action_nodes])
                if mx_ucb < 10000:
                    mx_nodes = [_ for _ in curr_state_node.child_action_nodes if _.ucb_score == mx_ucb]
                    action_node = choice(mx_nodes)
                else:
                    action_node = max(curr_state_node.child_action_nodes,
                                      key=lambda node: node.ucb_score)

                curr_state, reward, done, info = search_env.step(action_node.action)
                if curr_state == 1 or curr_state == 6:
                    reward = -10
                trajectory_rewards.append(reward)
                curr_depth += 1

                # check if one of the children has this sampled state
                if curr_state in [_.state for _ in action_node.possible_child_state_nodes]:
                    for child_state_node in action_node.possible_child_state_nodes:
                        if child_state_node.state == curr_state:
                            curr_state_node = child_state_node
                else:
                    # create state node
                    curr_state_node = self.create_state_node(action_node=action_node,
                                                             new_state=curr_state,
                                                             depth=curr_depth,
                                                             search_env=search_env,
                                                             done=done)
                    break

                if done or curr_depth >= self.max_depth:
                    break

            if not done and curr_depth < self.max_depth:
                # rollout
                leaf_rollout_return = 0
                leaf_rollout_depth = 0
                while not done and curr_depth < self.max_depth:
                    _rollout_state, reward, done, _ = search_env.step(search_env.action_space.sample())

                    print(f"rollout state: {_rollout_state}")
                    if _rollout_state == 1 or _rollout_state == 6:
                        print(f"see me")
                        reward = -100

                    # TODO | shouldn't gamma's exponent be the total depth, not just leaf_rollout_depth?
                    leaf_rollout_return += self.gamma ** leaf_rollout_depth * reward  # discounted
                    leaf_rollout_depth += 1
                    curr_depth += 1
                trajectory_rewards.append(leaf_rollout_return)

            # start from end of trajectory back till root
            reward_idx = len(trajectory_rewards) - 1
            discounted_return = 0
            # return of node is rewards from that node till leaf node,
            # plus return of leaf node, adjusted by the discount factor
            while action_node:  # backup the Monte carlo return till root

                discounted_return = self.gamma * discounted_return + trajectory_rewards[reward_idx]
                action_node.update_value(discounted_return)
                curr_state_node = action_node.parent_state_node
                curr_state_node.update_value()
                action_node = curr_state_node.parent_action_node
                reward_idx -= 1

        return self.root.most_visited_action_child.action


    def get_action(self, observation, env, prob_distribution):
        return self.compute_plan(env=env, starting_state=observation, prob_distribution=prob_distribution)

    def get_action_scores(self, observation, env, prob_distribution):
        action = self.compute_plan(env=env, starting_state=observation, prob_distribution=prob_distribution)
        action_scores = {}
        for child in self.root.child_action_nodes:
            action_scores[child.action] = {'value': child.value,
                                           'pamcts_value': child.ucb_score,
                                           'num_visits': child.visits,
                                           'policy_value': child.policy_value}
        return action_scores