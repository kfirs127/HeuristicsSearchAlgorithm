import random
from topspin import TopSpinState
from BWAS import BWAS
from heuristics import BaseHeuristic


def bellmanUpdateTraining(bellman_update_heuristic):
    pass


def bootstrappingTraining(bootstrapping_heuristic):
    n = bootstrapping_heuristic.get_n()
    k = bootstrapping_heuristic.get_k()
    num_of_states = 1024

    random_states = _get_k_states(TopSpinState(list(range(1, n+1)), k), num_of_states)
    random.shuffle(random_states)

    first_time = True
    basic_heuristic = BaseHeuristic(n, k).get_h_values

    T = 10000
    to_train = {}
    global_counter = 0

    for _ in range(5):
        global_counter += 1
        print_counter = 1
        for random_state in random_states:
            print(f"Processing state {print_counter} / {num_of_states} for run {global_counter}")
            print_counter += 1

            does_one_solved = False
            path = None

            while not does_one_solved and T <= 1000000:
                if first_time:
                    first_time = False
                    path, _ = BWAS(random_state, 5, 10, basic_heuristic, T)
                else:
                    path, _ = BWAS(random_state, 5, 10, bootstrapping_heuristic.get_h_values, T)
                if path is not None:
                    does_one_solved = True
                else:
                    T = max(T * 2, 1000000)

            # TODO if we changed the otfer ot path from start to end - need to change the logic here as well
            counter = 0
            if path is not None:
                for p in path[0]:
                    t_p = TopSpinState(p)
                    if t_p not in to_train or counter < to_train[t_p]:
                        to_train[t_p] = counter
                    counter += 1

        bootstrapping_heuristic.train_model(list(to_train.keys()), list(to_train.values()))


# state - goal_state
# rs - number of Random States
def _get_k_states(state, rs):
    random_states = []
    for _ in range(rs):
        random_state = state
        ml = random.randint(5, 40)
        for _ in range(ml):
            action_index = random.randint(0, 2)
            random_state = random_state.get_neighbors()[action_index]

        random_states.append(random_state)

    return random_states
