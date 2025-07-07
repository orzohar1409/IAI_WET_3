from copy import deepcopy
import numpy as np


class State:
    def __init__(self, i, j, reward, utility=0):
        self.i = i
        self.j = j
        self.reward = reward
        self.utility = utility

    def get_reward(self):
        return self.reward

    def set_utility(self, utility):
        self.utility = utility
    def get_quards(self):
        return self.i, self.j
    def __eq__(self, other):
        if not isinstance(other, State):
            return False
        return self.i == other.i and self.j == other.j
    def __hash__(self):
        return hash((self.i, self.j))
class MDPWrapper:
    def __init__(self, mdp):
        self.mdp = mdp
        self.num_row = mdp.num_row
        self.num_col = mdp.num_col
        self.terminal_states = mdp.terminal_states
        self.actions = mdp.actions
        self.transition_function = mdp.transition_function
        self.gamma = mdp.gamma
        self.board = [[
            State(i, j, mdp.board[i][j]) for j in range(mdp.num_col)
        ] for i in range(mdp.num_row)]

    def is_terminal(self, state):
        return state.get_quards() in self.terminal_states

    def is_wall(self, state):
        i, j = state.get_quards()
        return self.board[i][j].reward == 'WALL'
    def in_bounds(self, state):
        i, j = state.get_quards()
        return 0 <= i < self.num_row and 0 <= j < self.num_col and not self.is_wall(state)
    def get_all_states(self):
        for i in range(self.num_row):
            for j in range(self.num_col):
                yield self.board[i][j]

    def set_utility(self, utility):
        for i in range(self.num_row):
            for j in range(self.num_col):
                self.board[i][j].set_utility(utility[i][j])
    def get_utility(self):
        return [[self.board[i][j].utility for j in range(self.num_col)] for i in range(self.num_row)]

    def get_expected_utility(self, state, action):
        """
        Given a state and an action, return the expected utility of the next states.
        Assumes transition_function[(i, j)][action] = list of ((ni, nj), probability)
        """
        s_coords = state.get_quards()
        transitions = self.transition_function[s_coords][action]

        expected_util = 0.0
        for (ni, nj), prob in transitions:
            expected_util += prob * self.board[ni][nj].utility
        return expected_util



def value_iteration(mdp, U_init, epsilon=10 ** (-3)):
    """
    Given the MDP, an initial utility matrix U_init, and epsilon (upper bound on error from optimal utility),
    run the value iteration algorithm and return the utility matrix U at convergence.
    """
    U = deepcopy(U_init)
    mdp_wrapper = MDPWrapper(mdp)
    mdp_wrapper.set_utility(U)
    delta = float('inf')

    # Iterate until convergence threshold reached
    while delta >= (epsilon * (1 - mdp_wrapper.gamma)) / mdp_wrapper.gamma:
        delta = 0
        new_U = mdp_wrapper.get_utility()
        for state in mdp_wrapper.get_all_states():
            if mdp_wrapper.is_wall(state):
                continue
            if mdp_wrapper.is_terminal(state):
                new_U[state.i][state.j] = state.get_reward()
                continue

            max_util = float('-inf')
            for action in mdp_wrapper.actions:
                max_util = max(max_util, mdp_wrapper.get_expected_utility(state, action))

            new_value = state.get_reward() + mdp.gamma * max_util
            new_U[state.i][state.j] = new_value
            delta = max(delta, abs(new_value - U[state.i][state.j]))
        mdp_wrapper.set_utility(new_U)
    return mdp_wrapper.get_utility()


def get_policy(mdp, U):
    # TODO:
    # Given the mdp and the utility of each state - U (which satisfies the Belman equation)
    # return: the policy
    #
    mdp_wrapper = MDPWrapper(mdp)
    mdp_wrapper.set_utility(U)
    policy = [[None for _ in range(mdp.num_col)] for _ in range(mdp.num_row)]
    for curr_state in mdp_wrapper.get_all_states():
        if mdp_wrapper.is_wall(curr_state) or mdp_wrapper.is_terminal(curr_state):
            policy[curr_state.i][curr_state.j] = None
            continue
        best_action = None
        best_util = float('-inf')
        for a in mdp.actions:
            expected_util = mdp_wrapper.get_expected_utility(curr_state, a)
            if expected_util > best_util:
                best_util = expected_util
                best_action = a
        policy[curr_state.i][curr_state.j] = best_action
    return policy


def policy_evaluation(mdp, policy):
    """
    Given the MDP and a fixed policy, compute the utility U(s) for each state s
    using iterative policy evaluation.
    """
    mdp_wrapper = MDPWrapper(mdp)
    u_new = [[0.0 for _ in range(mdp_wrapper.num_col)] for _ in range(mdp_wrapper.num_row)]
    for state in mdp_wrapper.get_all_states():
        if mdp_wrapper.is_terminal(state):
            u_new[state.i][state.j] = state.get_reward()
            continue
        if mdp_wrapper.is_wall(state):
            u_new[state.i][state.j] = 0.0
            continue
        u_new[state.i][state.j] = state.get_reward() + mdp_wrapper.gamma * mdp_wrapper.get_expected_utility(state, policy[state.i][state.j])

    return u_new


def policy_iteration(mdp, policy_init):
    """
    Run policy iteration algorithm using the structure in the pseudocode.
    Returns the optimal policy.
    """
    mdp_wrapper = MDPWrapper(mdp)
    changed = True
    curr_policy = deepcopy(policy_init)
    while changed:
        changed = False
        U = policy_evaluation(mdp, curr_policy)
        mdp_wrapper.set_utility(U)
        new_policy = deepcopy(curr_policy)
        for state in mdp_wrapper.get_all_states():
            if mdp_wrapper.is_wall(state) or mdp_wrapper.is_terminal(state):
                new_policy[state.i][state.j] = None
                continue
            best_action = None
            best_util = float('-inf')
            for action in mdp.actions:
                expected_util = mdp_wrapper.get_expected_utility(state, action)
                if expected_util > best_util:
                    best_util = expected_util
                    best_action = action
            if curr_policy[state.i][state.j] != best_action:
                changed = True
                new_policy[state.i][state.j] = best_action

    return curr_policy


"""For this functions, you can import what ever you want """


def get_all_policies(mdp, U, epsilon=10 ** (-3)):  # You can add more input parameters as needed
    # TODO:
    # Given the mdp, and the utility value U (which satisfies the Belman equation)
    # print / display all the policies that maintain this value
    # (a visualization must be performed to display all the policies)
    #
    # return: the number of different policies
    #

    # ====== YOUR CODE: ======
    raise NotImplementedError
    # ========================


def get_policy_for_different_rewards(mdp, epsilon=10 ** (-3)):  # You can add more input parameters as needed
    # TODO:
    # Given the mdp
    # print / displas the optimal policy as a function of r
    # (reward values for any non-finite state)
    #

    # ====== YOUR CODE: ======
    raise NotImplementedError
    # ========================
