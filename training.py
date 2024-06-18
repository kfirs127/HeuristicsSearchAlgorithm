import random
from topspin import TopSpinState
from BWAS import BWAS


def bellmanUpdateTraining(bellman_update_heuristic):
    n = bellman_update_heuristic.get_n()
    k = bellman_update_heuristic.get_k()
    batch_size = 64

    to_train = {}

    for i in range(1000):
        random_states = _get_k_states(TopSpinState(list(range(1, n + 1)), k), batch_size)
        random.shuffle(random_states)

        print_counter = 1
        for random_state in random_states:
            if (print_counter - 1) % 100 == 0:
                print(f"Level {i + 1}, Processing state {print_counter - 1} / {batch_size}")
            print_counter += 1

            if random_state.is_goal():
                to_train[random_state] = 0
            else:
                successors = random_state.get_neighbors()
                min_cost = float('inf')
                for succ in successors:
                    if succ.is_goal():
                        min_cost = 1
                        break
                    else:
                        h_value = bellman_update_heuristic.get_h_values([succ])[0]
                        min_cost = min(min_cost, 1 + h_value)

                if random_state not in to_train or min_cost < to_train[random_state]:
                    to_train[random_state] = min_cost

        print(f'at level {i + 1} we solved {len(to_train)} times')
        bellman_update_heuristic.train_model(list(to_train.keys()), list(to_train.values()))


def bootstrappingTraining(bootstrapping_heuristic):
    n = bootstrapping_heuristic.get_n()
    k = bootstrapping_heuristic.get_k()
    batch_size = 128

    T = 10000
    T_max = 100000
    to_train = {}

    for i in range(300):
        print_counter = 1
        does_one_solved = False

        random_states = _get_k_states(TopSpinState(list(range(1, n + 1)), k), batch_size)
        random.shuffle(random_states)

        for random_state in random_states:
            if (print_counter - 1) % 100 == 0:
                print(f"Level {i + 1}, Processing state {print_counter - 1} / {batch_size}")
            print_counter += 1

            path, _ = BWAS(random_state, 5, 10, bootstrapping_heuristic.get_h_values, T)
            if path is not None:
                does_one_solved = True
                counter = len(path) - 1
                for p in path:
                    t_p = TopSpinState(p)
                    if t_p not in to_train or counter < to_train[t_p]:
                        to_train[t_p] = counter
                    counter -= 1

        if not does_one_solved:
            T = max(T * 2, T_max)

        print(f'at level {i + 1} we solved {len(to_train)} times')
        bootstrapping_heuristic.train_model(list(to_train.keys()), list(to_train.values()))


# state - goal_state
# rs - number of Random States
def _get_k_states(state, rs):
    random_states = []
    for _ in range(rs):
        random_state = state
        ml = random.randint(0, 40)
        for _ in range(ml):
            action_index = random.randint(0, 2)
            random_state = random_state.get_neighbors()[action_index]

        random_states.append(random_state)

    return random_states
