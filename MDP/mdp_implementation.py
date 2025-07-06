from copy import deepcopy
import numpy as np


def value_iteration(mdp, U_init, epsilon=10 ** (-3)):
    from copy import deepcopy

    U = deepcopy(U_init)
    rows, cols = mdp.num_row, mdp.num_col

    def is_terminal(s):
        return s in mdp.terminal_states

    def get_reward(s):
        i, j = s
        cell = mdp.board[i][j]
        if cell == "WALL":
            return 0
        return float(cell)

    def in_bounds(i, j):
        return 0 <= i < rows and 0 <= j < cols and mdp.board[i][j] != '#'

    def next_state(s, a):
        di, dj = mdp.actions[a]
        ni, nj = s[0] + di, s[1] + dj
        return (ni, nj) if in_bounds(ni, nj) else s

    delta = float('inf')
    gamma = mdp.gamma
    transition = mdp.transition_function

    while delta >= (epsilon * (1 - gamma)) / gamma:
        delta = 0
        new_U = deepcopy(U)

        for i in range(rows):
            for j in range(cols):
                s = (i, j)

                if mdp.board[i][j] == 'WALL' or is_terminal(s):
                    continue

                max_util = float('-inf')
                for a in mdp.actions:
                    probs = transition[a]
                    dirs = list(mdp.actions.keys())
                    next_states = [next_state(s, d) for d in dirs]
                    expected_util = 0
                    for p, ns in zip(probs, next_states):
                        expected_util += p * U[ns[0]][ns[1]]
                    max_util = max(max_util, expected_util)

                new_U[i][j] = get_reward(s) + gamma * max_util
                delta = max(delta, abs(new_U[i][j] - U[i][j]))

        U = new_U

    return U


def get_policy(mdp, U):
    # TODO:
    # Given the mdp and the utility of each state - U (which satisfies the Belman equation)
    # return: the policy
    #

    # ====== YOUR CODE: ======
    raise NotImplementedError
    # ========================


def policy_evaluation(mdp, policy):
    # TODO:
    # Given the mdp, and a policy
    # return: the utility U(s) of each state s
    #

    # ====== YOUR CODE: ======
    raise NotImplementedError
    # ========================


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
