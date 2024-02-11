from abc import ABC, abstractmethod

class Retrieve(ABC):

    def __init__(self): # 路径的设置和参数设置为主
        pass

    @abstractmethod
    def setup_retrieve(self): # load 机制
        pass

    @abstractmethod
    def search(self): # passages retrieval
        pass
