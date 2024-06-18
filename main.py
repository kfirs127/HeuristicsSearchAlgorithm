from training import bootstrappingTraining, bellmanUpdateTraining
from heuristics import BaseHeuristic, BellmanUpdateHeuristic, BootstrappingHeuristic
from BWAS import BWAS
from topspin import TopSpinState

def run():
    instance_1 = [1, 7, 10, 3, 6, 9, 5, 8, 2, 4, 11]  # easy instance
    instance_2 = [1, 5, 11, 2, 6, 3, 9, 4, 10, 7, 8]  # hard instance

    start1 = TopSpinState(instance_1, 4)
    # base_heuristic = BaseHeuristic(11, 4)
    # path, expansions = BWAS(start1, 5, 10, base_heuristic.get_h_values, 1000000)
    # if path is not None:
    #     print(expansions)
    #     for vertex in path:
    #         print(vertex)
    # else:
    #     print("unsolvable")
    #
    start2 = TopSpinState(instance_2, 4)
    # BU_heuristic = BaseHeuristic(11, 4)
    # path, expansions = BWAS(start2, 5, 10, BU_heuristic.get_h_values, 1000000)
    # if path is not None:
    #     print(expansions)
    #     for vertex in path:
    #         print(vertex)
    # else:
    #     print("unsolvable")


    # bootstrapping_heuristic = BootstrappingHeuristic(11, 4)
    # bootstrappingTraining(bootstrapping_heuristic)
    # path, expansions = BWAS(start1, 5, 10, bootstrapping_heuristic.get_h_values, 1000000)
    # if path is not None:
    #     print(expansions)
    #     for vertex in path:
    #         print(vertex)
    # else:
    #     print("unsolvable")
    #
    # path, expansions = BWAS(start2, 5, 10, bootstrapping_heuristic.get_h_values, 1000000)
    # if path is not None:
    #     print(expansions)
    #     for vertex in path:
    #         print(vertex)
    # else:
    #     print("unsolvable")


    bellman_heuristic = BellmanUpdateHeuristic(11, 4)
    bellmanUpdateTraining(bellman_heuristic)
    path, expansions = BWAS(start1, 5, 10, bellman_heuristic.get_h_values, 1000000)
    if path is not None:
        print(expansions)
        for vertex in path:
            print(vertex)
    else:
        print("unsolvable")

    path, expansions = BWAS(start2, 5, 10, bellman_heuristic.get_h_values, 1000000)
    if path is not None:
        print(expansions)
        for vertex in path:
            print(vertex)
    else:
        print("unsolvable")

if __name__ == '__main__':
    run()

