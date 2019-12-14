from abc import abstractmethod, ABCMeta


class BasePlotter(metaclass=ABCMeta):
    def __init__(self):
        pass

    @abstractmethod
    def plot(self, **kwargs):
        raise NotImplementedError()
