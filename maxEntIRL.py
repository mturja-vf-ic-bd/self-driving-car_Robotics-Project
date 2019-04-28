import numpy as np
from learning import IRL_helper # get the Reinforcement learner
from playing import play  # get the RL Test agent, gives out feature expectations after 2000 frames
from nn import neural_net  # construct the nn and send to playing

BEHAVIOR = 'bumping' # yellow/brown/red/bumping
FRAMES = 100000 # number of RL training frames per iteration of IRL

class RelEntIRL:
    def __init__(self, expert_demos, nonoptimal_demos, num_frames, behavior):
        self.expert_demos = expert_demos
        self.policy_features = nonoptimal_demos
        self.num_features = len(self.expert_demos[0])
        self.weights = np.zeros((self.num_features,))
        self.num_frames = num_frames
        self.behavior = behavior

    def calculate_objective(self):
        '''For the partition function Z($\theta$), we just sum over all the exponents of their rewards, similar to
        the equation above equation (6) in the original paper.'''
        objective = np.dot(self.expert_feature, self.weights)
        for i in range(self.policy_features.shape[0]):
            objective -= np.exp(np.dot(self.policy_features[i], self.weights))
        return objective

    def calculate_expert_feature(self):
        self.expert_feature = np.zeros_like(self.weights)
        for i in range(len(self.expert_demos)):
            self.expert_feature += self.expert_demos[i]
        self.expert_feature /= len(self.expert_demos)
        return self.expert_feature

    def train(self, step_size=1e-4, num_iters=50000, print_every=5000):
        self.calculate_expert_feature()
        importance_sampling = np.zeros((len(self.policy_features),))
        for i in range(num_iters):
            update = np.zeros_like(self.weights)
            for j in range(len(self.policy_features)):
                importance_sampling[j] = np.exp(np.dot(self.policy_features[j], self.weights))
            importance_sampling /= np.sum(importance_sampling, axis=0)
            weighted_sum = np.sum(np.multiply(np.array([importance_sampling, ] * self.policy_features.shape[1]).T, \
                                              self.policy_features), axis=0)
            self.weights += step_size * (self.expert_feature - weighted_sum)
            # One weird trick to ensure that the weights don't blow up.
            self.weights = self.weights / np.linalg.norm(self.weights, keepdims=True)
            if i % print_every == 0:
                print("Value of objective is: " + str(self.calculate_objective()))


    def getRLAgentFE(self, W, i, count=1):  # get the feature expectations of a new policy using RL agent
        nn_param = [164, 150]
        params = {
            "batchSize": 100,
            "buffer": 50000,
            "nn": nn_param,
            "observe": 1000
        }
        IRL_helper(W, self.behavior, self.num_frames, i, nn_param, params)  # train the agent and save the model in a file used below
        saved_model = 'saved-models_'+self.behavior+'/evaluatedPolicies/'+str(i)+'-164-150-'+str(params["batchSize"])+'-'+str(params["buffer"])+"-"+str(self.num_frames)+'.h5'  # use the saved model to get the FE
        model = neural_net(self.num_features, [164, 150], saved_model)
        if count > 1:
            feat = np.zeros((count, self.num_features))
            for a in range(count):
                feat[a, :] = play(model, W)
            return feat

        return play(model, W)  # return feature expectations by executing the learned policy


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
    saved_model = 'saved-models_' + behavior + '/evaluatedPolicies/' + str(i) + '-164-150-' + str(
    params["batchSize"]) + '-' + str(params["buffer"]) + "-" + str(n_frames) + '.h5'  # use the saved model to get the FE
    model = neural_net(n_feat, [164, 150], saved_model)

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


def generate_non_optimal_trajectories(behavior='red', count=5):
    try:
        with open(behavior + '_init_policy.pkl', 'rb') as f:
            total = pickle.load(f)
    except FileNotFoundError:
        Red = get_trajectory_policy('red', count)
        Brown = get_trajectory_policy('brown', count)
        Yellow = get_trajectory_policy('yellow', count)

        with open("random_feat.pkl", "rb") as f:
            randomPolicyFE = pickle.load(f)
            randomPolicyFE = randomPolicyFE[0: count, :]

        if behavior == 'red':
            total = np.vstack((Brown, Yellow, randomPolicyFE))
        elif behavior == 'yellow':
            total = np.vstack((Brown, Red, randomPolicyFE))
        elif behavior == 'brown':
            total = np.vstack((Yellow, Red, randomPolicyFE))

        with open(behavior + '_init_policy.pkl', 'wb') as f:
            pickle.dump(total, f)

    return total


import pickle
import time
if __name__ == "__main__":
    expertPolicyYellowFE = [[7.5366e+00, 4.6350e+00, 7.4421e+00, 3.1817e-01, 8.3398e+00, 1.3710e-08, 1.3419e+00,
                            0.0000e+00]]
    # ^feature expectations for the "follow Yellow obstacles" behavior
    expertPolicyRedFE = [[7.9100e+00, 5.3745e-01, 5.2363e+00, 2.8652e+00, 3.3120e+00, 3.6478e-06, 3.82276074e+00,
                         1.0219e-17]]
    # ^feature expectations for the follow Red obstacles behavior
    expertPolicyBrownFE = [[5.2210e+00, 5.6980e+00, 7.7984e+00, 4.8440e-01, 2.0885e-04, 9.2215e+00, 2.9386e-01,
                           4.8498e-17]]
    # ^feature expectations for the "follow Brown obstacles" behavior
    expertPolicyBumpingFE = [[7.5313e+00, 8.2716e+00, 8.0021e+00, 2.5849e-03, 2.4300e+01, 9.5962e+01, 1.5814e+01,
                             1.5538e+03]]
    # ^feature expectations for the "nasty bumping" behavior

    randomPolicyFE = generate_non_optimal_trajectories(BEHAVIOR, count=1)
    if BEHAVIOR == 'red':
        expert_policy = expertPolicyRedFE
    elif BEHAVIOR == 'yellow':
        expert_policy = expertPolicyYellowFE
    elif BEHAVIOR == 'bumping':
        expert_policy = expertPolicyBumpingFE
    elif BEHAVIOR == 'brown':
        expert_policy = expertPolicyBrownFE

    rl_iter = 1
    start = time.time()
    for i in range(0, rl_iter):
        relent = RelEntIRL(expert_policy, randomPolicyFE, FRAMES, BEHAVIOR)
        relent.train()
        print("weights: ", relent.weights)
        #learner_feature = relent.getRLAgentFE(relent.weights, 10000, count=2)
        # randomPolicyFE = learner_feature
        # randomPolicyFE = np.vstack((randomPolicyFE, learner_feature))
        # print("Learner feature at iteration ", i, ": ", learner_feature)
        # end = time.time()
        # print("Time: ", (end - start) // 60, " min ", (end - start) % 60, " second")

        with open("saved_weights_" + BEHAVIOR + "/iter_" + str(i) + "_weights.pkl", "wb") as f:
            pickle.dump(relent.weights, f)




