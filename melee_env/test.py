from enum import Enum
import random

class stage(Enum):
    bf = 0
    stadium = 1
    fd = 2

chosen = random.choice(list(stage))

class S():
    def __init__(self, i):
        self.a = i
        self.b = i + 1



d = [S(1), S(3), S(5)]

t = {a.a : a.b for a in d}
#print({t[b] for b in t})

a = {0: 0, 1: 1, 2: 0, 3: 1, 4: 0}

b = [1,2,3]

for e in [x for x in a if a[x] == 0]:
    print(e)

