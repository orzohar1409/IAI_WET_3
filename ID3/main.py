import numpy as np

import utils
def main():
    """
    This is the main function where the primary logic of the script resides.
    """
    features, train_dataset, test_dataset = utils.load_data_set('ID3')
    random_binary_vector = np.random.randint(2, size=len(features)).astype(str)

    print(random_binary_vector)
    print(np.ones_like(features))
    print(len(features))
    print(utils.accuracy(np.ones_like(features), random_binary_vector))

if __name__ == "__main__":
    # This block only executes when the script is run directly.
    main()