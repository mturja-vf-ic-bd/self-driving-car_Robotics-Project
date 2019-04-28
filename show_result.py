from playing import play  # get the RL Test agent, gives out feature expectations after 2000 frames
from nn import neural_net  # construct the nn and send to playing
import numpy as np


def get_trajectory_policy(behavior='red', count=5):
    nn_param = [164, 150]
    params = {
        "batchSize": 100,
        "buffer": 50000,
        "nn": nn_param
    }
    n_feat = 8
    n_frames = 100000
    if behavior == 'red':
        i = 3
    elif behavior == 'brown':
        i = 5
    elif behavior == 'yellow':
        i = 100
    elif behavior == 'bumping':
        i = 1
    saved_model = 'saved-models_' + behavior + '/evaluatedPolicies/' + str(i) + '-164-150-' + str(
    params["batchSize"]) + '-' + str(params["buffer"]) + "-" + str(n_frames) + '.h5'  # use the saved model to get the FE
    model = neural_net(n_feat, [164, 150], saved_model)

    W = np.zeros(8)

    if behavior == 'red':
        W = [0.2816, -0.5547, -0.2297,   0.6824, -0.3025, 0.0004,
         0.0525, -0.0075]
    elif behavior == 'brown':
        W = [-0.2627, 0.0363, 0.0931, 0.0046, -0.1829, 0.6987, -0.5922, -0.2201]
    elif behavior == 'yellow':
        W = [-0.0880, -0.0624, 0.0914, -0.0114, 0.6690, -0.0771, -0.6650, -0.2897]


    feat = np.zeros((count, n_feat))
    for a in range(count):
        feat[a, :] = play(model, W)
    return feat


import sys
if __name__ == '__main__':
    behavior = sys.argv[1]
    get_trajectory_policy(behavior)
