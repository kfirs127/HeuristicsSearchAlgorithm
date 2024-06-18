import heapq
import numpy as np


class Node:
    def __init__(self, s, g, p, f):
        self.s = s
        self.g = g
        self.p = p
        self.f = f

    def __lt__(self, other):
        return self.g < other.g if self.f == other.f else self.f < other.f


def BWAS(start, W, B, heuristic_function, T):
    OPEN = []
    CLOSED = {start: 0}
    UB, n_UB = np.inf, None
    LB = 0
    expentions = 0
    g = 0
    p = None
    f = heuristic_function([start])[0]
    N_start = Node(start, g, p, f)
    heapq.heappush(OPEN, N_start)

    while len(OPEN) > 0 and expentions <= T:
        generated = []
        batch_expanstions = 0
        while len(OPEN) > 0 and batch_expanstions < B and expentions <= T:
            n = heapq.heappop(OPEN)
            s, g, p, f = n.s, n.g, n.p, n.f
            expentions += 1
            batch_expanstions += 1
            if len(generated) == 0:
                LB = max(f, LB)
            if s.is_goal():
                if UB > g:
                    UB, n_UB = g, n
                continue
            for successor in s.get_neighbors():
                g_successor = g + 1  # c(s, s_) = 1
                if successor not in CLOSED or g_successor < CLOSED[successor]:
                    CLOSED[successor] = g_successor
                    generated.append((successor, g_successor, n))
        if LB >= UB:
            return path_to_goal(n_UB), expentions
        generated_states = [state for state, _, _ in generated]
        heuristics = heuristic_function(generated_states)
        for i in range(len(generated)):
            s, g, p = generated[i]
            h = heuristics[i]
            f = g + W * h
            n_successor = Node(s, g, p, f)
            heapq.heappush(OPEN, n_successor)

    return path_to_goal(n_UB), expentions


def path_to_goal(n_UB):
    if n_UB is None:
        return None

    path = []
    current_s, current_p = n_UB.s, n_UB.p
    while current_p is not None:
        path.append(current_s.get_state_as_list())
        current_s = current_p.s
        current_p = current_p.p
    path.append(current_s.get_state_as_list())
    path = path[::-1]
    return path
