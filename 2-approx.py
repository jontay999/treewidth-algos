from graph import Graph, TreeDecomposition
import sys

# Implements 2k + 1 - approximation of a graph's treewidth as demonstrated in https://arxiv.org/pdf/2104.07463.pdf.

NOT_VALID = sys.maxsize # Denotes invalid DP table entries

class TwoApproximator():

    def __init__(self,T: TreeDecomposition, r: int, k: int):
        self.T = T # The current tree decomposition
        self.r = r # The current pointer to a node in T
        self.k = k # Determines the factor of approximation


    def move(self,s: int):
        pass

    def split(self):
        pass

    def state(self):
        pass

    def edit(self, T1: TreeDecomposition, T2, TreeDecomposition, r: int, new_r: int):
        pass


    # Accesses the dynamic programming table U at the specified indices, with bounds checking
    def _get_dp_entry(self, h: int):
        if h < 0 or h > self.T.width():
            return NOT_VALID
        
        