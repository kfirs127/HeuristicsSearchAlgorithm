from collections import deque
import random


class TopSpinState:

    def __init__(self, state, k=4):
        self._n = len(state)
        self._k = k
        self._state = deque(state)

    def is_goal(self):
        return list(self._state) == list(range(1, self._n + 1))

    def get_state_as_list(self):
        return list(self._state)

    # we decided to take a random permutation of the possibale actions.
    def get_neighbors(self):
        return [self._flip_rotation(), self._clockwise_rotation(), self._counterclockwise_rotation()]

    def _clockwise_rotation(self):
        copy_state = self._state.copy()
        copy_state.rotate(1)
        return TopSpinState(list(copy_state))

    def _counterclockwise_rotation(self):
        copy_state = self._state.copy()
        copy_state.rotate(-1)
        return TopSpinState(list(copy_state))

    def _flip_rotation(self):
        copy_state = self._state.copy()
        flip_rotation = TopSpinState(list(copy_state)[:self._k][::-1] + list(copy_state)[self._k:])
        return flip_rotation

    def __repr__(self):
        return f"TopSpinState({list(self._state)})"

    def __eq__(self, other):
        if isinstance(other, TopSpinState):
            return self._state == other._state
        return False

    def __hash__(self):
        return hash(tuple(self._state))
