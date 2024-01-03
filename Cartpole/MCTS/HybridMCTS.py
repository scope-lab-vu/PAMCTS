from MCTS.MCTSNode import MCTSNode
import copy
from gym import spaces
import itertools
from envs.discrete_w_start import DiscreteWS

class HybridMCTS():

    def __init__(self,
                 gamma = 0.99,
                 learning_agent=None,
                 alpha=0.5,
                 num_iter=10000,
                 max_depth=500):

        self.num_iterations = num_iter
        self.max_depth = max_depth
        self.learning_agent = learning_agent
        self.gamma = gamma
        self.alpha = alpha
        self.root = MCTSNode(None,
                             None,
                             policy_value=None,
                             alpha=self.alpha)

    def combinations(self, space):
        if isinstance(space, DiscreteWS):
            return range(space.start, space.start + space.n) # support kserver
        if isinstance(space, spaces.Discrete):
            return range(space.n)
        elif isinstance(space, spaces.Tuple):
            return itertools.product(*[self.combinations(s) for s in space.spaces])
        else:
            raise NotImplementedError

    def get_plan_from_root(self):
        """
        from self.root, find the sequence of actions by accessing node.best_child
        i.e greedy policy w.r.t value function
        """

        root = self.root
        plan = []
        # while root.best_child:
        #     plan.append(root.best_child.action)
        #     root = root.best_child

        # TODO | update with node ucb score instead? How do we choose best child?
        # while root.most_visited_child:
            # visited_children = [child for child in root.children if child.visits > 0]
            # best_ucb_child = max(visited_children, key=lambda child : child.ucb_score)
            # plan.append(best_ucb_child.action)
            # root = best_ucb_child

        # TODO | Should have random if tied
        # TODO | Update with most visits
        while root.most_visited_child:
            plan.append(root.most_visited_child.action)
            root = root.most_visited_child

        return plan

    def update_tree(self, action):
        """
        update the tree with an action. this means an action has been
        executed in the real-world, now the mcts root node will change
        to one of its children so we can re-use the mcts tree.
        """
        # child_actions = [child.action for child in self.root.children]
        # if action in child_actions:
        #     self.root = self.root.children[child_actions.index(action)]
        #     self.root.parent = None
        # else:
        #     self.root = MCTSNode(None, None) # TODO | update with alpha and such

        # TODO | try just reseting root each time. Doesn't allow tree reuse, but in stochastic environment that's
        # TODO | probably fine
        self.root = MCTSNode(None, None, policy_value=None, alpha=self.alpha)

    @staticmethod
    def evaluate_mcts_policy(root_node, env, max_depth=500, render=True):

        """
        greedy evaluation of mcts policy
        args:
            root_node: the root noded of mcts tree
            env: the openai gym environment
            max_depth: maximum depth to simulate
            render: whether to render the evaluation
        """

        cum_reward = 0
        env_backup = copy.deepcopy(env)
        done = False
        depth = 0

        while root_node and root_node.best_child and not done and depth < max_depth:

            best_action = root_node.best_child.action
            _, r, done, info = env_backup.step(best_action)
            cum_reward += r
            depth += 1
            root_node = root_node.best_child
            if render:
                env_backup.render()

        while not done and depth < max_depth: # entering unexplored region, take random actions
            random_action = env_backup.action_space.sample()
            _, r, done, _ = env_backup.step(random_action)
            cum_reward += r
            depth+= 1
            if render:
                env_backup.render()

        env_backup.close()
        print(f'planner achieved cumulative return of {cum_reward}')


    def compute_plan(self, env, starting_state):

        """
        compute a MCTS plan by executing num_iterations rollouts with max_depth.
        returns list of actions found by MCTS (open-loop control sequence)
        """

        policy_values_ledger = []

        for iteration in range(self.num_iterations):

            node = self.root
            env_mcts = copy.deepcopy(env)
            depth = 0
            # if iteration % 100 == 0:
            #     print(f'performing iteration {iteration} of MCTS')

            done = False
            trajectory_rewards = []  # store the individual reward along the trajectory.
            curr_state = copy.deepcopy(starting_state)

            while node.children:
                node = max(node.children, key=lambda node: node.ucb_score)

                curr_state, reward, done, info = env_mcts.step(node.action)  # assume deterministic environment
                trajectory_rewards.append(reward)
                depth+=1
                if done:
                    print('done')
                if done or depth >= self.max_depth:
                    break

            # at this point we are either at a leaf node or at a terminal state
            if not done and depth < self.max_depth: # it's a leaf node. let's add its children
                policy_values = self.learning_agent.get_q_values(curr_state)
                policy_values_ledger.append(policy_values)
                node.children = [MCTSNode(node, a, policy_value=policy_values[a], alpha=self.alpha) for a in self.combinations(env_mcts.action_space)]

                ####
                ## TODO | ava debug - trying the

                # rollout with a random policy till we reach a terminal state
                leaf_rollout_return = 0
                leaf_rollout_depth = 0

                while not done and depth < self.max_depth:

                    # TODO | Ava Debug
                    _rollout_state, reward, done, _ = env_mcts.step(env_mcts.action_space.sample()) # TODO greedy rollout for k-server?
                    # _rollout_state, reward, done, _ = env_mcts.step(0)

                    # TODO | shouldn't gamma's exponent be the total depth, not just leaf_rollout_depth?
                    leaf_rollout_return += self.gamma ** leaf_rollout_depth * reward  # discounted
                    leaf_rollout_depth += 1
                    depth+=1

                if done:
                    print('')
                # append the Monte carlo return of the leaf node to trajectory reward.
                trajectory_rewards.append(leaf_rollout_return)

            # start from end of trajectory back till root
            reward_idx = len(trajectory_rewards) - 1
            discounted_return = 0
            # return of node is rewards from that node till leaf node,
            # plus return of leaf node, adjusted by the discount factor
            while node:  # backup the Monte carlo return till root

                discounted_return = self.gamma * discounted_return + trajectory_rewards[reward_idx]
                node.update_value(discounted_return)

                # move to parent node
                reward_idx -= 1
                node = node.parent

        # print(max([abs(_[0] - _[1]) for _ in policy_values_ledger]))

        return self.get_plan_from_root()

    def get_action(self, observation, env):
        self.update_tree(None)
        action_sequence = self.compute_plan(env=env,
                                            starting_state=observation)

        return action_sequence[0]

    def get_action_scores(self, observation, env):
        self.update_tree(None)
        action_sequence = self.compute_plan(env=env,
                                            starting_state=observation)

        action_scores = {}
        for child in self.root.children:
            action_scores[child.action] = {'value': child.value,
                                           'pamcts_value': child.ucb_score,
                                           'num_visits': child.visits,
                                           'policy_value': child.policy_value}
        return action_scores




