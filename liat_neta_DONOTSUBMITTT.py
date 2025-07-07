from copy import deepcopy
import numpy as np


def value_iteration(mdp, U_init: list[list[float]], epsilon=10 ** (-3)):
    # Given the mdp, the initial utility of each state - U_init,
    #   and the upper limit - epsilon.
    # run the value iteration algorithm and
    # return: the U obtained at the end of the algorithms' run.
    #

    # ====== YOUR CODE: ======
    U = deepcopy(U_init)

    for (i, j) in mdp.terminal_states:
        reward = float(mdp.board[i][j])
        U[i][j] = reward

    while True:
        delta = 0
        U_prev = deepcopy(U)
        for i in range(mdp.num_row):
            for j in range(mdp.num_col):
                if mdp.board[i][j] == "WALL" or (i, j) in mdp.terminal_states:
                    continue

                reward = float(mdp.board[i][j])
                max_util = -np.inf
                for a in mdp.actions:
                    util_expectation = 0
                    for idx, actual_action in enumerate(mdp.actions.keys()):
                        p = mdp.transition_function[a][idx]
                        next_state = mdp.step((i, j), actual_action)
                        util_expectation += p * U_prev[next_state[0]][next_state[1]]
                    max_util = max(max_util, util_expectation)
                U[i][j] = reward + mdp.gamma * max_util

                if abs(U[i][j] - U_prev[i][j]) > delta:
                    delta = abs(U[i][j] - U_prev[i][j])

        if delta < epsilon * (1 - mdp.gamma) / mdp.gamma:
            break
    return U
    # ========================


def get_policy(mdp, U):
    # Given the mdp and the utility of each state - U (which satisfies the Belman equation)
    # return: the policy
    #

    # ====== YOUR CODE: ======
    policy = [[None for col in range(mdp.num_col)] for row in range(mdp.num_row)]

    for i in range(mdp.num_row):
        for j in range(mdp.num_col):
            if mdp.board[i][j] == "WALL" or (i, j) in mdp.terminal_states:
                continue
            reward = float(mdp.board[i][j])
            max_util = -np.inf
            for a in mdp.actions:
                util = 0
                for idx, actual_action in enumerate(mdp.actions.keys()):
                    p = mdp.transition_function[a][idx]
                    next_state = mdp.step((i, j), actual_action)
                    util += p * (reward + mdp.gamma * U[next_state[0]][next_state[1]])

                if util > max_util:
                    max_util = util
                    policy[i][j] = a
    return policy
    # ========================


def policy_evaluation(mdp, policy):
    # Given the mdp, and a policy
    # return: the utility U(s) of each state s
    #

    # ====== YOUR CODE: ======
    n = mdp.num_col * mdp.num_row
    P = np.zeros((n, n))
    I = np.eye(n)
    R = np.zeros((n, 1))

    for (i, j) in mdp.terminal_states:
        reward = float(mdp.board[i][j])
        s = i * mdp.num_col + j
        R[s][0] = reward

    for i in range(mdp.num_row):
        for j in range(mdp.num_col):
            if mdp.board[i][j] == "WALL" or (i, j) in mdp.terminal_states:
                continue
            s = i * mdp.num_col + j
            reward = float(mdp.board[i][j])
            R[s][0] = reward

            a = policy[i][j]
            for idx, actual_action in enumerate(mdp.actions.keys()):
                p = mdp.transition_function[a][idx]
                next_state = mdp.step((i, j), actual_action)
                s_tag = next_state[0] * mdp.num_col + next_state[1]
                P[s][s_tag] += p

    U_vector = np.linalg.inv(I - mdp.gamma * P) @ R
    U = [[U_vector[i * mdp.num_col + j].item() for j in range(mdp.num_col)] for i in range(mdp.num_row)]
    return U
    # ========================


def policy_iteration(mdp, policy_init):
    # Given the mdp, and the initial policy - policy_init
    # run the policy iteration algorithm
    # return: the optimal policy
    #

    # ====== YOUR CODE: ======
    policy = deepcopy(policy_init)
    while True:
        U = policy_evaluation(mdp, policy)
        unchanged = True
        for i in range(mdp.num_row):
            for j in range(mdp.num_col):
                if mdp.board[i][j] == "WALL" or (i, j) in mdp.terminal_states:
                    continue
                curr_expectation = 0
                pi = policy[i][j]
                for idx, actual_action in enumerate(mdp.actions.keys()):
                    p = mdp.transition_function[pi][idx]
                    next_state = mdp.step((i, j), actual_action)
                    curr_expectation += p * U[next_state[0]][next_state[1]]

                for a in mdp.actions:
                    expectation = 0
                    for idx, actual_action in enumerate(mdp.actions.keys()):
                        p = mdp.transition_function[a][idx]
                        next_state = mdp.step((i, j), actual_action)
                        expectation += p * U[next_state[0]][next_state[1]]
                    if expectation > curr_expectation:
                        curr_expectation = expectation
                        policy[i][j] = a
                        unchanged = False
        if unchanged:
            break
    return policy
    # ========================


"""For this functions, you can import what ever you want """


def get_all_policies(mdp, U, epsilon=10 ** (-3)):  # You can add more input parameters as needed
    # Given the mdp, and the utility value U (which satisfies the Belman equation)
    # print / display all the policies that maintain this value
    # (a visualization must be performed to display all the policies)
    #
    # return: the number of different policies
    #

    # ====== YOUR CODE: ======
    policies = [["" for col in range(mdp.num_col)] for row in range(mdp.num_row)]
    amount_policies = 1

    for i in range(mdp.num_row):
        for j in range(mdp.num_col):
            if mdp.board[i][j] == "WALL" or (i, j) in mdp.terminal_states:
                policies[i][j] = None
                continue
            reward = float(mdp.board[i][j])

            counter = 0
            for a in mdp.actions:
                util = 0
                for idx, actual_action in enumerate(mdp.actions.keys()):
                    p = mdp.transition_function[a][idx]
                    next_state = mdp.step((i, j), actual_action)
                    util += mdp.gamma * p * U[next_state[0]][next_state[1]]

                if abs(util + reward - U[i][j]) < epsilon:
                    policies[i][j] += a[0]
                    counter += 1
            amount_policies *= counter

    if amount_policies > 0:
        mdp.print_policy(policies)
    return amount_policies
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
