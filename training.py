import random
from topspin import TopSpinState
from BWAS import BWAS


def bellmanUpdateTraining(bellman_update_heuristic):
    n = bellman_update_heuristic.get_n()
    k = bellman_update_heuristic.get_k()
    batch_size = 128

    global_counter = 1
    to_train = {}

    for _ in range(64):
        random_states = _get_k_states(TopSpinState(list(range(1, n + 1)), k), batch_size)
        random.shuffle(random_states)

        print_counter = 1
        for random_state in random_states:
            print(f"Level {global_counter}, Processing state {print_counter} / {batch_size}")
            print_counter += 1

            successors = random_state.get_neighbors()
            min_cost = float('inf')
            for succ in successors:
                if succ.is_goal():
                    cost = 1
                else:
                    h_value = bellman_update_heuristic.get_h_values([succ])[0]
                    cost = 1 + h_value
                if cost < min_cost:
                    min_cost = cost
            to_train[random_state] = min_cost

        print(f'at level {global_counter} we solved {len(to_train)} times')
        global_counter += 1
        bellman_update_heuristic.train_model(list(to_train.keys()), list(to_train.values()))


def bootstrappingTraining(bootstrapping_heuristic):
    n = bootstrapping_heuristic.get_n()
    k = bootstrapping_heuristic.get_k()
    batch_size = 128

    T = 10000
    T_max = 100000
    global_counter = 1
    to_train = {}

    for _ in range(8): # 128 per batch * 32 iterations = 4096 examples
        print_counter = 1
        does_one_solved = False

        random_states = _get_k_states(TopSpinState(list(range(1, n + 1)), k), batch_size)
        random.shuffle(random_states)

        for random_state in random_states:
            print(f"Level {global_counter}, Processing state {print_counter} / {batch_size}")
            print_counter += 1
            counter = 0

            path, _ = BWAS(random_state, 5, 10, bootstrapping_heuristic.get_h_values, T)
            # TODO if we changed the order ot path from start to end - need to change the logic here as well
            if path is not None:
                does_one_solved = True
                for p in path[0]:
                    t_p = TopSpinState(p)
                    if t_p not in to_train or counter < to_train[t_p]:
                        to_train[t_p] = counter
                    counter += 1

        if not does_one_solved:
            T = max(T * 2, T_max)

        print(f'at level {global_counter} we solved {len(to_train)} times')
        global_counter += 1
        bootstrapping_heuristic.train_model(list(to_train.keys()), list(to_train.values()))


# state - goal_state
# rs - number of Random States
def _get_k_states(state, rs):
    random_states = []
    for _ in range(rs):
        random_state = state
        ml = random.randint(5, 20)
        for _ in range(ml):
            action_index = random.randint(0, 2)
            random_state = random_state.get_neighbors()[action_index]

        random_states.append(random_state)

    return random_states
