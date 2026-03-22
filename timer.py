import time

class timer:
    def __init__(self, description):
        self.description = description
    
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end = time.perf_counter()
        print(f"{self.description} Elapsed: {self.end - self.start:.6f}s")
