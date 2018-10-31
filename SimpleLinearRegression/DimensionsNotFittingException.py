class DimensionsNotFittingException(Exception):
    
    def __init__(self, message = "Dimensions of training data and labels are not matching!"):
        self.message = message
    def display_message(self):
        print(self.message)

