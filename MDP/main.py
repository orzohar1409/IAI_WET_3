import argparse
import os
from mdp import MDP
from mdp_implementation import value_iteration, get_policy, policy_evaluation, policy_iteration, get_all_policies,get_policy_for_different_rewards


def is_valid_file(parser, arg):
    if not os.path.exists(arg):
        parser.error("The file %s does not exist!" % arg)


def example_driver():

    """
    This is an example of a driver function
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("Board", help="A file that holds the board and the reward for each state")
    parser.add_argument("TerminalStates", help="A file that contains the terminal states in the board")
    parser.add_argument("TransitionFunction", help="A file that contains the transition function")
    args = parser.parse_args()

    board = args.Board
    terminal_states = args.TerminalStates
    transition_function = args.TransitionFunction

    is_valid_file(parser, board)
    is_valid_file(parser, terminal_states)

    board_env = []
    with open(board, 'r') as f:
        for line in f.readlines():
            row = line[:-1].split(',')
            board_env.append(row)

    terminal_states_env = []
    with open(terminal_states, 'r') as f:
        for line in f.readlines():
            row = line[:-1].split(',')
            terminal_states_env.append(tuple(map(int, row)))

    transition_function_env = {}
    with open(transition_function, 'r') as f:
        for line in f.readlines():
            action, prob = line[:-1].split(':')
            prob = prob.split(',')
            transition_function_env[action] = tuple(map(float, prob))

    # initialising the env
    mdp = MDP(board=board_env,
              terminal_states=terminal_states_env,
              transition_function=transition_function_env,
              gamma=0.9)
    #
    # print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
    # print("@@@@@@ The board and rewards @@@@@@")
    # print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
    # mdp.print_rewards()
    #
    # print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
    # print("@@@@@@@@@ Value iteration @@@@@@@@@")
    # print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
    #
    # U = [[0, 0, 0, 0],
    #      [0, 0, 0, 0],
    #      [0, 0, 0, 0]]
    #
    # print("\nInitial utility:")
    # mdp.print_utility(U)
    # print("\nFinal utility:")
    # U_new = value_iteration(mdp, U)
    # mdp.print_utility(U_new)
    # print("\nFinal policy:")
    # policy = get_policy(mdp, U_new)
    # mdp.print_policy(policy)
    #
    # print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
    # print("@@@@@@@@@ Policy iteration @@@@@@@@")
    # print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
    #
    # print("\nInitial policy:")
    # policy = [['RIGHT', 'RIGHT', 'RIGHT', None],
    #           ['UP', None, 'UP', None],
    #           ['UP', 'LEFT', 'UP', 'LEFT']]
    # print("\nPolicy evaluation:")
    # U_eval = policy_evaluation(mdp, policy)
    # mdp.print_utility(U_eval)
    #
    # policy = [['UP', 'UP', 'UP', None],
    #           ['UP', None, 'UP', None],
    #           ['UP', 'UP', 'UP', 'UP']]
    #
    # print("\nInitial policy:")
    # mdp.print_policy(policy)
    # print("\nFinal policy:")
    # policy_new = policy_iteration(mdp, policy)
    # mdp.print_policy(policy_new)
    #
    # print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
    # print("@@@@@@@@@ get_all_policies @@@@@@@@")
    # print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
    # # Get U from U_get_all_policies file
    # U = [[0.424, 0.579, 0.734, 1.0],
    #     [0.311, 0, 0.332, -1.0],
    #     [0.198, 0.116, 0.176, -0.069]]
    # print("get_all_policies:")
    # print(get_all_policies(mdp, U))
    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
    print("@@@@@@@@@ get_policy_for_different_rewards @@@@@@@@")
    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')

    print(get_policy_for_different_rewards(mdp))
    print("Done!")


if __name__ == '__main__':

    # run our example
    example_driver()
