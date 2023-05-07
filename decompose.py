# Implementation of algorithm described in https://epubs.siam.org/doi/pdf/10.1137/S0097539793251219
# A LINEAR-TIME ALGORITHM FOR FINDING TREE-DECOMPOSITIONS OF SMALL TREEWIDTH*

from UndirectedGraph import UndirectedGraph
from TreeDecomposition import TreeDecomposition
from typing import Tuple, Union, Set
from networkx.algorithms.approximation.treewidth import *
from utils import *

# Returns a treewidth and a tree decomposition using networkx
def treewidth(G: UndirectedGraph) -> Tuple[int, TreeDecomposition]:
    # uses heuristics but run it 20 times Monte Carlo style 
    tw, td, mapping = float('inf'), None, None
    
    for _ in range(20):
        G_i, mapping_i = G.randomize()
        nx_i = G_i.convert_to_nx()

        # Use min degree heuristic
        tw_i, td_i = treewidth_min_degree(nx_i)
        if(tw_i < tw):
            tw, td, mapping = tw_i, td_i, mapping_i

        # use min fill in heuristic
        tw_i, td_i =  treewidth_min_fill_in(nx_i)
        if(tw_i < tw):
            tw, td, mapping = tw_i, td_i, mapping_i
    
    treedec = TreeDecomposition(G)
    for bags in td.nodes:
        for b in bags:
            treedec.add_to_bag(mapping[b], treedec.next_bag)
        treedec.increase_bag()
    return tw, treedec

def decompose(G: UndirectedGraph, k: int) -> Union[bool, TreeDecomposition]:
    """
    G: UndirectedGraph
    k: expected tree width

    returns either
    - false -> treewidth(G) > k
    - tree decomp of G with treewwidth <= 4k

    """

    num_e, num_v = len(G.edge_list), G.size

    # note: this number is huge, small graphs will only hit one branch of the control flow
    c1 = 4*k**2 + 12*k + 16 # just a constant
    c2 = 1 / (8*k**2 + 24 *k + 32)
    d = 2*k**3 * (k+1) * c1

    # first condition because it is a default upperbound and the second condition comes from Lemma 2.3
    if k <= num_v-1 and num_e > k * num_v - (k * (k+1))//2:
        return False
    
    # when nodes reach O(1) apply networkx heuristic to get a tree decomposition
    if(num_v < 15):
        tw, td = treewidth(G)
        if(tw > k): return False
        return td
    
    # In typical smaller graphs, num_friendly_vertices = num_vertices
    num_friendly_vertices = 0
    vertex_degrees = G.vertex_degrees()
    
    # low_degrees[v] true if vertex v has low degree
    low_degrees = {v: deg < d for v, deg in vertex_degrees.items()}

    for v, is_low_degree in low_degrees.items():
        if not is_low_degree: continue
        for neigh in G.edges[v]:
            if not low_degrees[neigh]: continue
            num_friendly_vertices += 1
            break
    
    # note that the first condition will typically be true for small graphs because c1 >> num_v
    if (num_friendly_vertices > (num_v / c1)):
        """
        Main idea:
        - find maximal matching in G
        - compute graph G' = (V', E') by contracting every edge in M
        - recursively apply algo to G'
        - if G' has treewidth > k -> G treewidth > k also (refer to Lemma 3.4)
        - use tree decomp of G' with width k to build tree decomp of G with width <= 2k+1
        """

        matching = G.maximal_matching()
        
        # This produces a minor of the original graph, G'
        G_prime, new_edges = G.contract_graph(matching)
        new_edge_mapping = {v:u for u,v in new_edges.items()}
        contracted_edge_mapping = {v: u for u,v in matching}

        # yields a tree decomposition of G_prime of k width
        result = decompose(G_prime, k) # returns a TreeDecomposition Object
        
        # Lemma 3.4
        if result is False: 
            return False
        
        # Lemma 3.3 to reconstruct a graph with < 2k+1  width
        treedec = TreeDecomposition(G)
        for v, bags in result.vertex_bags.items():
            original_v = new_edge_mapping[v]
            for b in bags:
                treedec.add_to_bag(original_v, b)
        
        for retained_v, contracted_v in contracted_edge_mapping.items():
            # add contracted vertex to all bags with the retained vertex 
            for bag in treedec.vertex_bags[retained_v]:
                treedec.add_to_bag(contracted_v, bag)
        
        # opted to neglect Theorem 2.4: given a tree decomposition of size l, determine if tree decomposition of size k exists
        return treedec
        
    
    else:
        # improving graph does not affect treewidth
        G_prime = compute_improved_graph(G, k)
        i_simplicial_vertices = compute_i_simplicial_vertices(G_prime)
        degrees_prime = G_prime.vertex_degrees()

        # if there exists I-simplicial vertex with degree at least k+1, means improved graph have clique with k+2 -> treewidth > k
        for v, deg in degrees_prime.items():
            if deg >= k+1: return False

        # means treewidth larger than k
        if len(i_simplicial_vertices) < c2 * num_v: return False

        # apply modifications to a copy of G
        G_copy = G.copy()
        for v in i_simplicial_vertices:
            G_copy.remove_node(v)
        
        # recursively apply
        result = decompose(G_copy, k)
        
        # since G_copy is subgraph, treewidth of G_copy > k, means treewidth G > k.
        if result is False:
            return False
        
        # Result = (X,T) , Tree Decomposition
        # For all i-simplicial vertices called v, all of v' neighbors are a subset of a bag, b in  X
        # add a new node v' to T with the bag containing v' (b') = v + v's neighbors, and make b', adjacent to b
        # such a bag b exists by lemma 4.8

        # duplicate the old treedec
        treedec = TreeDecomposition(G)
        for v, bags in result.vertex_bags.items():
            for b in bags:
                treedec.add_to_bag(v, b)

        # add back I-simplicial vertices in the treedec
        for v in i_simplicial_vertices:
            found_bag = False
            neighbors = G.edges[v] - i_simplicial_vertices
            for bag, bag_vertices in result.bags.items():
                if len(bag_vertices.intersection(neighbors)) == len(neighbors):
                    next_bag = treedec.next_bag
                    treedec.add_to_bag(v,next_bag)
                    for neigh in G.edges[v]:
                        treedec.add_to_bag(neigh, next_bag)
                    treedec.increase_bag()
                    found_bag = True
                    break
            
            if not found_bag:
                raise Exception("Lemma 4.8 has failed us!")
            
        return treedec


def test_graph_functions():
    random.seed(34)   
    g1 = generateRandomGraph(10,0.4)
    matching = g1.maximal_matching()
    contracted_graph, matching = g1.contract_graph(matching)
    assert matching == {1: 1, 4: 2, 6: 3, 7: 4, 8: 5, 9: 6, 10: 7}
    improved_graph = compute_improved_graph(g1, 2)
    assert improved_graph == g1


if __name__ == "__main__":
    test_simplicial_graph() 
    test_graph_functions()
    random.seed(34)
    test_treewidth(decompose, False, 4)
