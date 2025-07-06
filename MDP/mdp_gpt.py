from copy import deepcopy
import numpy as np

def value_iteration(mdp, U_init, epsilon=10 ** (-3)):
    """
    Given the MDP, an initial utility matrix U_init, and epsilon (upper bound on error from optimal utility),
    run the value iteration algorithm and return the utility matrix U at convergence.
    """
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

    # Iterate until convergence threshold reached
    while delta >= (epsilon * (1 - gamma)) / gamma:
        delta = 0
        new_U = deepcopy(U)
        for i in range(rows):
            for j in range(cols):
                s = (i, j)
                if mdp.board[i][j] == 'WALL' or is_terminal(s):
                    # No update for wall or terminal state
                    continue
                # Compute the Bellman update for state s
                max_util = float('-inf')
                for a in mdp.actions:
                    # Expected utility for taking action a at state s
                    probs = transition[a]
                    dirs = list(mdp.actions.keys())
                    next_states = [next_state(s, d) for d in dirs]
                    expected_util = 0
                    for p, ns in zip(probs, next_states):
                        expected_util += p * U[ns[0]][ns[1]]
                    if expected_util > max_util:
                        max_util = expected_util
                # Bellman equation: immediate reward + gamma * expected future utility
                new_U[i][j] = get_reward(s) + gamma * max_util
                delta = max(delta, abs(new_U[i][j] - U[i][j]))
        U = new_U
    return U

def get_policy(mdp, U):
    """
    Given the MDP and a utility matrix U (satisfying the Bellman equation),
    return a policy matrix mapping each state to an optimal action (one of mdp.actions),
    using an arbitrary optimal action if multiple exist.
    Terminal states and walls will be denoted with 0 and 'WALL', respectively.
    """
    rows, cols = mdp.num_row, mdp.num_col
    policy = [[None for _ in range(cols)] for __ in range(rows)]
    for i in range(rows):
        for j in range(cols):
            s = (i, j)
            cell = mdp.board[i][j]
            if cell == 'WALL':
                policy[i][j] = 'WALL'
            elif s in mdp.terminal_states:
                policy[i][j] = 0  # no action for terminal state
            else:
                best_action = None
                best_util = float('-inf')
                # Evaluate each action's expected utility
                for a in mdp.actions:
                    probs = mdp.transition_function[a]
                    dirs = list(mdp.actions.keys())
                    # Determine next states for each possible direction outcome
                    next_states = []
                    for d in dirs:
                        ni, nj = i + mdp.actions[d][0], j + mdp.actions[d][1]
                        if 0 <= ni < rows and 0 <= nj < cols and mdp.board[ni][nj] != '#':
                            next_states.append((ni, nj))
                        else:
                            next_states.append((i, j))
                    expected_util = 0.0
                    for p, ns in zip(probs, next_states):
                        expected_util += p * U[ns[0]][ns[1]]
                    if expected_util > best_util:
                        best_util = expected_util
                        best_action = a
                policy[i][j] = best_action
    return policy

def policy_evaluation(mdp, policy, epsilon=10 ** (-3)):
    """
    Given the MDP and a policy (mapping states to actions),
    return the utility matrix U for that policy (solution of Bellman equations for fixed policy).
    """
    rows, cols = mdp.num_row, mdp.num_col
    U = [[0.0 for _ in range(cols)] for __ in range(rows)]
    gamma = mdp.gamma
    # Determine convergence threshold
    threshold = (epsilon * (1 - gamma) / gamma) if gamma != 1 else epsilon
    while True:
        delta = 0.0
        new_U = deepcopy(U)
        for i in range(rows):
            for j in range(cols):
                s = (i, j)
                cell = mdp.board[i][j]
                if cell == 'WALL' or s in mdp.terminal_states:
                    new_U[i][j] = 0.0
                    continue
                a = policy[i][j]
                if a is None or a == 0 or a == 'WALL':
                    continue
                # Compute expected utility for the fixed policy's action at state s
                probs = mdp.transition_function[a]
                dirs = list(mdp.actions.keys())
                next_states = []
                for d in dirs:
                    ni, nj = i + mdp.actions[d][0], j + mdp.actions[d][1]
                    if 0 <= ni < rows and 0 <= nj < cols and mdp.board[ni][nj] != '#':
                        next_states.append((ni, nj))
                    else:
                        next_states.append((i, j))
                expected_util = 0.0
                for p, ns in zip(probs, next_states):
                    expected_util += p * U[ns[0]][ns[1]]
                # Immediate reward for current state
                reward = 0.0
                if cell != "WALL":
                    reward = float(cell)
                new_U[i][j] = reward + gamma * expected_util
                delta = max(delta, abs(new_U[i][j] - U[i][j]))
        U = new_U
        if delta < threshold:
            break
    return U

def policy_iteration(mdp, policy_init, epsilon=10 ** (-3)):
    """
    Given the MDP and an initial policy, run the policy iteration algorithm to find an optimal policy.
    Returns the optimal policy matrix.
    """
    rows, cols = mdp.num_row, mdp.num_col
    policy = deepcopy(policy_init)
    while True:
        # Policy evaluation for current policy
        U = policy_evaluation(mdp, policy, epsilon)
        policy_stable = True
        # Policy improvement: update policy by choosing best action for each state
        for i in range(rows):
            for j in range(cols):
                s = (i, j)
                cell = mdp.board[i][j]
                if cell == 'WALL' or s in mdp.terminal_states:
                    continue
                best_action = None
                best_value = float('-inf')
                for a in mdp.actions:
                    probs = mdp.transition_function[a]
                    dirs = list(mdp.actions.keys())
                    next_states = []
                    for d in dirs:
                        ni, nj = i + mdp.actions[d][0], j + mdp.actions[d][1]
                        if 0 <= ni < rows and 0 <= nj < cols and mdp.board[ni][nj] != '#':
                            next_states.append((ni, nj))
                        else:
                            next_states.append((i, j))
                    expected_util = 0.0
                    for p, ns in zip(probs, next_states):
                        expected_util += p * U[ns[0]][ns[1]]
                    reward = 0.0
                    if cell != "WALL":
                        reward = float(cell)
                    value = reward + mdp.gamma * expected_util
                    if value > best_value:
                        best_value = value
                        best_action = a
                if best_action is not None and policy[i][j] != best_action:
                    policy[i][j] = best_action
                    policy_stable = False
        if policy_stable:
            break
    return policy

def get_all_policies(mdp, U, epsilon=10 ** (-3)):
    """
    Given the MDP and an optimal utility matrix U, display all the policies that yield this utility.
    A visualization of the grid is printed where each state shows all optimal actions (e.g., 'UR' if both Up and Right are optimal).
    Returns the number of distinct optimal policies.
    """
    rows, cols = mdp.num_row, mdp.num_col
    optimal_actions = [[[] for _ in range(cols)] for __ in range(rows)]
    for i in range(rows):
        for j in range(cols):
            s = (i, j)
            cell = mdp.board[i][j]
            if cell == 'WALL' or s in mdp.terminal_states:
                optimal_actions[i][j] = []
            else:
                # Compute Q-values for each action
                q_values = {}
                for a in mdp.actions:
                    probs = mdp.transition_function[a]
                    dirs = list(mdp.actions.keys())
                    next_states = []
                    for d in dirs:
                        ni, nj = i + mdp.actions[d][0], j + mdp.actions[d][1]
                        if 0 <= ni < rows and 0 <= nj < cols and mdp.board[ni][nj] != '#':
                            next_states.append((ni, nj))
                        else:
                            next_states.append((i, j))
                    exp_util = 0.0
                    for p, ns in zip(probs, next_states):
                        exp_util += p * U[ns[0]][ns[1]]
                    reward = 0.0
                    if cell != "WALL":
                        reward = float(cell)
                    q_values[a] = reward + mdp.gamma * exp_util
                max_q = max(q_values.values())
                # Include all actions whose Q is within epsilon of max_q
                for a, q in q_values.items():
                    if abs(q - max_q) <= epsilon:
                        optimal_actions[i][j].append(a)
    # Print visualization of all optimal actions for each state
    for i in range(rows):
        row_display = []
        for j in range(cols):
            if mdp.board[i][j] == 'WALL':
                row_display.append("WALL")
            elif (i, j) in mdp.terminal_states:
                row_display.append("0")
            else:
                acts = optimal_actions[i][j]
                if not acts:
                    row_display.append(" ")
                elif len(acts) == 1:
                    row_display.append(acts[0])
                else:
                    # Concatenate first letters of multiple optimal actions
                    initials = "".join(sorted([a[0] for a in acts]))
                    row_display.append(initials)
        print("\t".join(row_display))
    # Compute number of distinct optimal policies (product of choices in each state)
    num_policies = 1
    for i in range(rows):
        for j in range(cols):
            if mdp.board[i][j] == 'WALL' or (i, j) in mdp.terminal_states:
                continue
            count = max(1, len(optimal_actions[i][j]))
            num_policies *= count
    return num_policies

def get_policy_for_different_rewards(mdp, epsilon=10 ** (-3)):
    """
    Given the MDP, vary the reward R for any non-terminal state and display how the optimal policy changes as a function of R.
    Returns a list of R values at which the optimal policy changes.
    """
    # Define a range of R values to test (assuming no changes beyond certain bounds)
    r_min, r_max = -1.0, 1.0
    step = 0.01
    prev_policy = None
    change_points = []
    policies = []
    original_board = deepcopy(mdp.board)
    # Identify non-terminal state positions
    non_terminal_positions = [(i, j) for i in range(mdp.num_row) for j in range(mdp.num_col)
                               if mdp.board[i][j] != 'WALL' and (i, j) not in mdp.terminal_states]
    r = r_min
    while r <= r_max + 1e-9:
        # Set reward of all non-terminal states to r
        for (i, j) in non_terminal_positions:
            mdp.board[i][j] = str(r)
        # Compute optimal policy for this reward value
        U_init = [[0.0 for _ in range(mdp.num_col)] for __ in range(mdp.num_row)]
        U_opt = value_iteration(mdp, U_init, epsilon)
        policy = get_policy(mdp, U_opt)
        if prev_policy is None:
            prev_policy = policy
            policies.append((r, policy))
        else:
            # Check if policy changed compared to previous
            policy_diff = any(policy[i][j] != prev_policy[i][j] for i in range(mdp.num_row) for j in range(mdp.num_col))
            if policy_diff:
                change_points.append(round(r, 2))
                policies.append((r, policy))
                prev_policy = policy
        r = round(r + step, 2)
    # Restore original board rewards
    mdp.board = original_board
    # Display each distinct policy and the range of R for which it applies
    if policies:
        # Policy for lowest R range
        first_threshold = change_points[0] if change_points else r_max
        print(f"Optimal policy for R < {first_threshold}:")
        mdp.print_policy(policies[0][1])
        for idx, r_val in enumerate(change_points):
            # r_val is a threshold where policy changes at that value
            next_threshold = change_points[idx+1] if idx+1 < len(change_points) else None
            current_policy = policies[idx+1][1] if idx+1 < len(policies) else policies[idx][1]
            if next_threshold is not None:
                print(f"\nOptimal policy for {r_val} <= R < {next_threshold}:")
            else:
                print(f"\nOptimal policy for R >= {r_val}:")
            mdp.print_policy(current_policy)
    return change_points
