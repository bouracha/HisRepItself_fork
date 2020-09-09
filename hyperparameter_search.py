import os
import argparse
import random
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument('--num_trials', type=int, default='10',
                    help='number of trials of randomly selected hyperparameters')
parser.add_argument('--epoch', type=int, default=50, help='number of epochs')

opt = parser.parse_args()


def random_hyperparameters():
    lambda_log_range = (2.302585092994046, -13.815510557964274)  # (10, 0.00001)
    dropout_range = (0.0, 0.3)

    lambda_ = np.e ** (random.uniform(lambda_log_range[0], lambda_log_range[1]))
    dropout = random.uniform(dropout_range[0], dropout_range[1])

    return lambda_, dropout

for i in range(opt.num_trials):
    (lambda_, dropout) = random_hyperparameters()
    command = 'python3 main_h36m_3d.py --kernel_size 10 --dct_n 20 --input_n 50 --output_n 10 --skip_rate 1 --batch_size 32 --test_batch_size 32 --in_features 66 --actions '+str(walking)+' --p_drop '+str(dropout)+' --lambda_ '+str(lambda_)
    print(command)
    os.system(command)
