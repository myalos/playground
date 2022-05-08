import random
import numpy as np

class A:
    def __init__(self):
        self.data = [9,4 ,3 ,7 ,5]

    def __getitem__(self, ind):
        return self.data[ind]

    def __iter__(self):
        while 1:
            self.idx = list(range(len(self.data)))
            random.shuffle(self.idx)
            for i in self.idx:
                yield self.__getitem__(i)


a = A()

print(np.arange(-2, 0))
print(np.arange(1, 3))

x = 0.5 * np.concatenate((np.arange(-2, 0), np.arange(1, 3)))

print(x)

y = np.arange(7)
print(y.shape)
print(y[:, np.newaxis].shape)
z = y[:, np.newaxis] - x
print(z.shape)
print(x)
print(y)
print(z)

print(np.argmin(np.abs(z), axis = 0))


