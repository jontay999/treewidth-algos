# Implementation of algorithm described in https://epubs.siam.org/doi/pdf/10.1137/S0097539793251219
# A LINEAR-TIME ALGORITHM FOR FINDING TREE-DECOMPOSITIONS OF SMALL TREEWIDTH*

import random
"""
Step 1:
apply O(n^2) algorithm that outputs either
-  treewidth of G > k
-  tree decomposition with  width <= 4k

Step 2: use graph minors

- G is minor of H if G can be obtained via vertex/edge deletion or edge contractions
- every graph has finite set of graphs that is closed under minors called obstruction set
- graph belongs to this set iff it has no graph from the obstruction set as a minor
- class of graphs with treewidth <= k is closed under minors for every fixed value of k

Check if the input graph is like that
- using dynamic programming in linear time
"""

"""
Algorithm terms

for some d fixed later
- low-degree vertex:  degree <= d
- high-degree vertex:  degree > d
- friendly vertex: is low-degree, and adjacent to another low-degree vertex
- simplicial vertex: neighbors form a clique
- improved graph: obtained by adding edges between all vertices that have at least k+1 common neighbors of degree <= k
- I-simplicial: simplicial in the improved graph of G and has degree <= k in G


"""

from graph import UndirectedGraph, generateRandomGraph


def decompose(G: UndirectedGraph, k: int):
    """
    G: UndirectedGraph
    k: expected tree width

    returns either
    - false -> treewidth(G) > k
    - tree decomp of G with treewwidth <= 4k

    """
    print(f"k: {k}\n")
    print(G)

    # check if |E|
    num_e, num_v = len(G.edge_list), G.size

    # note: this number is huge, not relevant for small graphs
    c1 = (4*k**2 + 12*k + 16) # just a constant
    d = 2*k**3 * (k+1) * c1

    # Lemma 2.3
    if num_e <= k * num_v - (k * (k+1))//2:
        return False
    
    num_friendly_vertices = 0
    vertex_degrees = G.vertex_degrees()
    
    print("Degrees:", vertex_degrees)
    # low_degrees[i] true if ith vertex has low degree
    low_degrees = [deg < d for deg in vertex_degrees]

    for i in range(num_v):
        if(low_degrees[i]):
            for neigh in G.edges[i+1]:
                if(low_degrees[neigh-1]):
                    num_friendly_vertices += 1
                    break
    
    # In typical smaller graphs, num_friendly_vertices = num_vertices
    print("Num friendly vertices:", num_friendly_vertices)
    
    # note that the first condition will typically be true for small graphs because c1 >> num_v
    if(num_friendly_vertices > (num_v / c1)):
        print("Maximal matching time!")
        # find maximal matching in G
        # compute graph G' = (V', E') by contracting every edge in M
        # recursively apply algo to G'
        # if G' has treewidth > k -> G treewidth > k also (refer to Lemma 3.4)
    else:
        print("Compute improved graph time!")

        # compute improved graph of G
        # if an I-simplicial vertex with degree > k+1 : stop

    

    
random.seed(34)
g1 = generateRandomGraph(10,25)

k_guess = 2
result = decompose(g1, k_guess)
if result is False:
    print(f"Treewidth of graph > {k_guess}")

