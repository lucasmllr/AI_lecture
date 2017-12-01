from queue import PriorityQueue
from copy import deepcopy

class BetterPriorityQueue():

    def __init__(self):
        self.queue = PriorityQueue()
        self.counter = 0
        self.elements = []

    def add(self, priority, element):
        count = deepcopy(self.counter)
        self.queue.put((priority, count, element))
        self.elements.append(element)
        self.counter += 1
        return

    def pop(self):
        _, _, elem = self.queue.get()
        self.elements.remove(elem)
        return elem

    def isin(self, element):
        return (element in self.elements)

    def empty(self):
        return self.queue.empty()

Q = BetterPriorityQueue()
Q.add(1, 2)
Q.add(1, 3)
print(Q.isin(4))