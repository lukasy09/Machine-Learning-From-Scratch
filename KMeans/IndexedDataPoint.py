import matplotlib.pyplot as plt
import numpy

class IndexedDataPoint:

    def __init__(self, data_point):
        self.cluster_index = None
        self.data = data_point

    def assign(self, cluster_index):
        self.cluster_index = cluster_index
