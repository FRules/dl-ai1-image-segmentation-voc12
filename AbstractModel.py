from abc import ABC, abstractmethod

from keras import Model


class AbstractModel(ABC):
    @abstractmethod
    def get_model(self, **kwargs) -> Model:
        pass
