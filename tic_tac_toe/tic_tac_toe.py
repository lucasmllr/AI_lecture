import numpy as np
from copy import deepcopy
import math
from queue import PriorityQueue
from betterPriorityQueue import BetterPriorityQueue

class Game_state():
    """class to represent the field a state of the game"""

    win_indices = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [0, 3, 6], [1, 4, 7], [2, 5, 8], [2, 4, 6], [0, 4, 8]]

    def __init__(self, player, tree, depth, parent=None, field_vec=np.zeros(9)):
        self.vec = field_vec
        self.player = player
        self.parent = parent
        self.tree = tree
        self.depth = depth
        tree.node_count += 1
        self.value = None
        self.next_move = None
        self.children = []
        self.leaf_count = 0
        self.is_leaf = self.is_leaf()
        if not self.is_leaf:
            self.find_children()
            tree.leaf_count += 1

    def is_leaf(self):
        winner = self.is_win()
        if winner != None:
            print('found win leaf', self.vec)
            self.value = winner
            self.queue_up_parent()
            return True
        elif 0 not in self.vec:
            print('found dead leaf', self.vec)
            self.value = 0
            self.queue_up_parent()
            return True
        else:
            return False

    def queue_up_parent(self):
        #print('about to queue up', self.parent.vec)
        if not self.tree.in_queue(self.parent) and self.parent!=None:
            print('queuing up', self.parent.vec, 'with priority', 9 - self.parent.depth)
            self.tree.add_to_queue(9 - self.parent.depth, self.parent)
            #print('added')
        return

    def is_win(self):
        vec = self.vec
        for set in self.win_indices:
            if vec[set[0]] != 0 and vec[set[0]] == vec[set[1]] \
                    and vec[set[1]] == vec[set[2]]:
                return vec[set[0]]
        return None

    def next_player(self):
        if self.player == 1:
            return -1
        return 1

    def find_children(self):
        for index, value in enumerate(self.vec):
            if value == 0:
                vec = deepcopy(self.vec)
                vec[index] = self.player
                new_child = Game_state(player=self.next_player(), tree=self.tree, parent=self, depth=self.depth + 1, \
                                       field_vec=vec)
                self.children.append(new_child)

class Game_tree():
    """class holding game tree and performing backpropagation and so..."""

    def __init__(self, player_at_move=1, field=np.zeros(9)):
        self.node_count = 1
        self.leaf_count = 0
        self.queue = BetterPriorityQueue()
        self.tree = Game_state(player=player_at_move, depth=7, tree=self, field_vec=field)

    def add_to_queue(self, priority, element):
        self.queue.add(priority, element)
        return

    def in_queue(self, element):
        return self.queue.isin(element)

    def backprop(self):
        current_depth = 9
        processed = 0
        while not self.queue.empty():
            processing = self.queue.pop()
            depth = processing.depth
            processed += 1
            if processed == 10000:
                print('processed another 10000 nodes')
                processed = 0
            if depth < current_depth:
                print('arrived at depth', depth)
                current_depth = depth
            best_value = float('inf')
            best_move = None
            if processing.player == -1:
                for child in processing.children:
                    if child.value < best_value:
                        best_value = child.value
                        best_move = child
            else:
                for child in processing.children:
                    if child.value > best_value:
                        best_value = child.value
                        best_move = child
            processing.value = best_value
            processing.next_move = best_move
            processing.queue_up_parent()
        return



def num_leaves():
    return np.sum([math.factorial(9)/math.factorial(i) for i in range(1, 9)])

field = np.array([1, 1, -1, 1, 0, 0, -1, -1, 1])
print('initializing game tree')
game = Game_tree(player_at_move=-1, field=field)
print('game initialized')
print('starting backpropagation')
game.backprop()
print('backpropagation finished')