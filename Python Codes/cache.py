import os
import pickle
from typing import Callable, List, Literal, Tuple, Any

class Cache:
    def __init__(self, cache_path):
        self.cache_path = cache_path

    def save_data(self, data):
        with open(f"{self.cache_path}.pickle", 'wb') as file:
            pickle.dump(data, file)

    def load_data(self):

        if os.path.exists(f"{self.cache_path}.pickle"):
            with open(f"{self.cache_path}.pickle", "rb") as file:
                data = pickle.load(file)
            return data
        
        else:
            return None

    
    def get_data(self, func: Callable, **kwargs: Any):

        loaded_data = self.load_data() 

        if loaded_data is None:

            func(**kwargs)

            return loaded_data