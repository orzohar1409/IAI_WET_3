from copy import deepcopy
import numpy as np


class State:
    def __init__(self, i, j, reward, utility=0):
        self.i = i
        self.j = j
        self.reward = reward
        self.utility = utility

    def get_reward(self):
        if self.reward == 'WALL':
            return 0
        return float(self.reward)

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
        This uses self.mdp.step to calculate valid resulting states.
        """
        expected_util = 0.0
        s_coords = state.get_quards()
        transition_probs = self.transition_function[action]  # tuple of 4 probabilities

        directions = self.actions.keys()
        for dir_name, prob in zip(directions, transition_probs):
            resulting_state_coords = self.mdp.step(s_coords, dir_name)
            ni, nj = resulting_state_coords
            next_state_utility = self.board[ni][nj].utility
            expected_util += prob * next_state_utility

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
            delta = max(delta, abs(new_value - state.utility))
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
    using linear algebra: U = (I - gamma * P)^-1 * R
    """
    mdp_wrapper = MDPWrapper(mdp)
    states = list(mdp_wrapper.get_all_states())
    state_to_index = {s: idx for idx, s in enumerate(states)}
    n = len(states)

    P = np.zeros((n, n))
    R = np.zeros((n, 1))
    gamma = mdp_wrapper.gamma

    for s in states:
        idx = state_to_index[s]
        R[idx][0] = s.get_reward()

        if mdp_wrapper.is_wall(s) or mdp_wrapper.is_terminal(s):
            continue

        a = policy[s.i][s.j]
        transition_probs = mdp.transition_function[a]

        for dir_name, prob in zip(mdp.actions.keys(), transition_probs):
            ni, nj = mdp.step((s.i, s.j), dir_name)
            next_state = mdp_wrapper.board[ni][nj]
            next_idx = state_to_index[next_state]
            P[idx][next_idx] += prob

    I = np.eye(n)
    solution = np.linalg.solve(I - gamma * P, R)

    # Convert back to 2D grid
    U_grid = [[0.0 for _ in range(mdp_wrapper.num_col)] for _ in range(mdp_wrapper.num_row)]
    for s, idx in state_to_index.items():
        U_grid[s.i][s.j] = round(solution[idx, 0], 6)

    return U_grid


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
        curr_policy = new_policy

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

    def convert_action(action):
        if action == 'UP':
            return "U"
        if action == 'DOWN':
            return "D"
        if action == 'LEFT':
            return "L"
        if action == 'RIGHT':
            return "R"

    mdp_wrapper = MDPWrapper(mdp)
    mdp_wrapper.set_utility(U)

    policies_per_state = [[[] for _ in range(mdp.num_col)] for _ in range(mdp.num_row)]
    for curr_state in mdp_wrapper.get_all_states():
        if mdp_wrapper.is_wall(curr_state) or mdp_wrapper.is_terminal(curr_state):
            continue
        for action in mdp.actions:
            expected_util = curr_state.get_reward() + mdp_wrapper.gamma * mdp_wrapper.get_expected_utility(curr_state, action)
            if abs(expected_util - curr_state.utility) < epsilon:
                policies_per_state[curr_state.i][curr_state.j].append(action)

    num_policies = 1
    for state in mdp_wrapper.get_all_states():
        if mdp_wrapper.is_wall(state) or mdp_wrapper.is_terminal(state):
            continue
        num_policies *= len(policies_per_state[state.i][state.j])


    policy = [["" for _ in range(mdp.num_col)] for _ in range(mdp.num_row)]
    for state in mdp_wrapper.get_all_states():
        if mdp_wrapper.is_wall(state):
            policy[state.i][state.j] = "WALL"
            continue
        if mdp_wrapper.is_terminal(state):
            policy[state.i][state.j] = str(state.get_reward())
            continue
        policy[state.i][state.j] = ' '.join(convert_action(a) for a in policies_per_state[state.i][state.j])
    mdp.print_policy(policy)
    return num_policies



def get_policy_for_different_rewards(mdp, epsilon=1e-3):
    """
    For each reward value r in [-5.00, 5.00] with step 0.01, set it as the reward
    for all non-terminal, non-wall states and compute the optimal policy.
    Whenever the policy changes from the previous one, display it and the reward range.
    """
    prev_policy = None
    r_values = np.round(np.arange(-5.00, 5.01, 0.01), 2)
    change_points = [-5.00]  # Start with the first change point
    mdp_wrapper = MDPWrapper(mdp)
    for r in r_values:
        # Set reward r to all non-terminal, non-wall states
        for state in mdp_wrapper.get_all_states():
            if mdp_wrapper.is_wall(state) or mdp_wrapper.is_terminal(state):
                continue
            state.reward = r


        U_init = [[0.0 for _ in range(mdp.num_col)] for _ in range(mdp.num_row)]
        mdp_wrapper.set_utility(U_init)
        # Compute the optimal policy for the current reward
        init_policy = [['UP', 'UP', 'UP', None],
              ['UP', None, 'UP', None],
              ['UP', 'UP', 'UP', 'UP']]
        policy = policy_iteration(mdp, policy_init=init_policy)
        # Compare with previous policy
        if prev_policy is None or policy != prev_policy:
            change_points.append(r)
            print(f"{change_points[-2]}<R(s)<{change_points[-1]}")
            mdp.print_policy(policy)
            prev_policy = policy

    return change_points

