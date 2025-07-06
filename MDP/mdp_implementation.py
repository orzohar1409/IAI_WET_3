from copy import deepcopy
import numpy as np

class State:
    def __init__(self, i, j, prob):
        self.i = i
        self.j = j
        self.prob = prob
class Utils:
    def __init__(self, mdp):
        self.mdp = mdp

    def is_terminal(self, s):
        return s in self.mdp.terminal_states

    def get_reward(self, s):
        i, j = s
        cell = self.mdp.board[i][j]
        if cell == "WALL":
            return 0
        return float(cell)
    def in_bounds(self, i, j):
        return 0 <= i < self.mdp.num_row and 0 <= j < self.mdp.num_col and self.mdp.board[i][j] != 'WALL'

    # get next states for a given state and direction, return all next states with their probabilities
    def next_states(self, s, a):
        mdp = self.mdp
        rows, cols = mdp.num_row, mdp.num_col
        i, j = s
        probs = mdp.transition_function[a]
        dirs = list(mdp.actions.keys())

        # Determine next states for each possible direction outcome
        next_states = []
        for d in dirs:
            ni, nj = i + mdp.actions[d][0], j + mdp.actions[d][1]
            state = State(ni, nj, probs[dirs.index(d)])
            if 0 <= ni < rows and 0 <= nj < cols and mdp.board[ni][nj] != 'WALL':
                next_states.append(state)
            else:
                next_states.append(State(i, j, probs[dirs.index(d)]))
        return next_states

def value_iteration(mdp, U_init, epsilon=10 ** (-3)):
    """
    Given the MDP, an initial utility matrix U_init, and epsilon (upper bound on error from optimal utility),
    run the value iteration algorithm and return the utility matrix U at convergence.
    """
    U = deepcopy(U_init)
    rows, cols = mdp.num_row, mdp.num_col
    utils = Utils(mdp)

    delta = float('inf')
    gamma = mdp.gamma
    transition = mdp.transition_function

    # Iterate until convergence threshold reached
    while delta >= (epsilon * (1 - gamma)) / gamma:
        delta = 0
        new_U = deepcopy(U)
        for i in range(rows):
            for j in range(cols):
                s = (i, j)
                if mdp.board[i][j] == 'WALL':
                    # No update for wall
                    continue
                if utils.is_terminal(s):
                    # Terminal state, utility is its reward
                    new_U[i][j] = utils.get_reward(s)
                    continue
                # Compute the Bellman update for state s
                max_util = float('-inf')
                for a in mdp.actions:
                    # Expected utility for taking action a at state s
                    next_states = utils.next_states(s, a)
                    expected_util = 0
                    for ns in next_states:
                        expected_util += ns.prob * U[ns.i][ns.j]
                    max_util = max(max_util, expected_util)
                # Bellman equation: immediate reward + gamma * expected future utility
                new_U[i][j] = utils.get_reward(s) + gamma * max_util
                delta = max(delta, abs(new_U[i][j] - U[i][j]))
        U = new_U
    return U


def get_policy(mdp, U):
    # TODO:
    # Given the mdp and the utility of each state - U (which satisfies the Belman equation)
    # return: the policy
    #

    rows, cols = mdp.num_row, mdp.num_col
    policy = [[None for _ in range(cols)] for __ in range(rows)]
    for i in range(rows):
        for j in range(cols):
            s = (i, j)
            cell = mdp.board[i][j]
            if cell == 'WALL':
                policy[i][j] = None
                continue
            if s in mdp.terminal_states:
                policy[i][j] = None
                continue

            best_action = None
            best_util = float('-inf')
            # Evaluate each action's expected utility
            for a in mdp.actions:
                # Get next states and their probabilities for this action
                next_states = Utils(mdp).next_states(s, a)

                # Calculate expected utility for this action
                expected_util = 0.0
                for ns in next_states:
                    expected_util += ns.prob * U[ns.i][ns.j]
                if expected_util > best_util:
                    best_util = expected_util
                    best_action = a
            policy[i][j] = best_action
    return policy


def policy_evaluation(mdp, policy):
    """
    Given the MDP and a fixed policy, compute the utility U(s) for each state s
    using iterative policy evaluation.
    """
    rows, cols = mdp.num_row, mdp.num_col
    U = [[0.0 for _ in range(cols)] for _ in range(rows)]
    gamma = mdp.gamma
    utils = Utils(mdp)

    delta = 0.0
    new_U = deepcopy(U)

    for i in range(rows):
        for j in range(cols):
            s = (i, j)
            cell = mdp.board[i][j]

            if cell == 'WALL':
                new_U[i][j] = 0.0
                continue
            if utils.is_terminal(s):
                new_U[i][j] = utils.get_reward(s)
                continue

            expected_util = 0.0
            next_states = Utils(mdp).next_states(s, policy[i][j])
            for ns in next_states:
                expected_util += ns.prob * U[ns.i][ns.j]

            reward = float(cell)
            new_U[i][j] = reward + gamma * expected_util

    return new_U


def policy_iteration(mdp, policy_init):
    # TODO:
    # Given the mdp, and the initial policy - policy_init
    # run the policy iteration algorithm
    # return: the optimal policy
    #

    # ====== YOUR CODE: ======
    raise NotImplementedError
    # ========================



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
