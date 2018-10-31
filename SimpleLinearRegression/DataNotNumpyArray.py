class DataNotNumpyArray(Exception):
    
    def __init__(self, msg = "Inputs data are not numpy.ndarray"):
        self.msg = msg
    def display_message(self):
        print(self.msg)