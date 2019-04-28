"""
Manually control the agent to provide expert trajectories.
The main aim is to get the feature expectaitons respective to the expert trajectories manually given by the user
Use the arrow keys to move the agent around
Left arrow key: turn Left
right arrow key: turn right
up arrow key: dont turn, move forward
down arrow key: exit 

Also, always exit using down arrow key rather than Ctrl+C or your terminal will be tken over by curses
"""
from flat_game import carmunk
import numpy as np
from nn import neural_net
import curses # for keypress

NUM_STATES = 8
N_FEAT = 8
GAMMA = 0.9 # the discount factor for RL algorithm


def play(screen, random_move=False):
    car_distance = 0
    weights = [1, 1, 1, 1, 1, 1, 1, 1]# just some random weights, does not matter in calculation of the feature expectations
    game_state = carmunk.GameState(weights)
    _, state, __ = game_state.frame_step((2))
    featureExpectations = np.zeros(N_FEAT)
    Prev = np.zeros(len(weights))
    while True:
        car_distance += 1

        if random_move:
            action = np.random.choice([0, 1], p=[0.5, 0.5])
        else:
            event = screen.getch()
            if event == curses.KEY_LEFT:
                action = 1
            elif event == curses.KEY_RIGHT:
                action = 0
            elif event == curses.KEY_DOWN:
                break
            else:
                action = 2

        # Take action. 
        #start recording feature expectations only after 100 frames
        immediateReward , state, readings = game_state.frame_step(action)
        if car_distance > 100:
            featureExpectations += (GAMMA**(car_distance-101))*np.array(readings)
            
        
        # Tell us something.
        changePercentage = (np.linalg.norm(featureExpectations - Prev)*100.0)/np.linalg.norm(featureExpectations)

        # print(car_distance)
        # print("percentage change in Feature expectation ::", changePercentage)
        Prev = np.array(featureExpectations)

        if car_distance % 1500 == 0:
            break

    return featureExpectations


if __name__ == "__main__":
    screen = curses.initscr()
    curses.noecho()
    curses.curs_set(0)
    screen.keypad(1)
    screen.addstr("Play the game")
    rand_sample = 100
    random_feat = np.zeros((rand_sample, N_FEAT))
    for i in range(0, rand_sample):
        print("Iteration: ", i)
        result = play(screen, True)
        random_feat[i, :] = result[:]
        print(result)

    import pickle
    with open("random_feat.pkl", "wb") as f:
        pickle.dump(random_feat, f)

