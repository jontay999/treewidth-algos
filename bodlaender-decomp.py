# Implementation of algorithm described in https://epubs.siam.org/doi/pdf/10.1137/S0097539793251219
# A LINEAR-TIME ALGORITHM FOR FINDING TREE-DECOMPOSITIONS OF SMALL TREEWIDTH*
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

from graph import UndirectedGraph, generateRandomGraph, TreeDecomposition
from typing import Union, List, Set, Tuple, Dict
import random
import networkx as nx
from networkx.algorithms.approximation import treewidth_min_degree, treewidth_min_fill_in

def compute_i_simplicial_vertices(G: UndirectedGraph) -> Set[int]:
    i_simplicial_vertices = set()
    vertex_set = G.vertices
    for v in vertex_set:
        neighbors = G.edges[v]
        non_neighbors = vertex_set - neighbors - set([v])
        for u in non_neighbors:
            if G.subgraph(neighbors | set([u])).is_simplicial():
                i_simplicial_vertices.add(v)
                break
    return i_simplicial_vertices

def compute_improved_graph(G: UndirectedGraph, k:int) -> UndirectedGraph:
    """
    https://arxiv.org/pdf/1304.6321.pdf for construction of improved graph
    Definition 2.5: Given a graph G = (V, E) and an integer k, the improved graph of G, denoted
    G_i , is obtained by adding an edge between each pair of vertices with at least k + 1 common
    neighbors of degree at most k in G.
    """
    G_i  = G.copy()
    degrees = G.vertex_degrees()
    vertex_pairs = []
    vertices = list(G.vertices)
    num_v = len(vertices)
    for i in range(num_v):
        for j in range(i+1, num_v):
            u,v = vertices[i], vertices[j]
            if u > v: u,v = v,u
            if v in G.edges[u]: continue # existing edges ignored
            common_neighbors = G.edges[u].intersection(G.edges[v])
            
            num_good_neighbors = len(common_neighbors)
            
            # ensure good neighbors have deg <= k
            for neigh in common_neighbors:
                if degrees[neigh] <= k: continue
                num_good_neighbors -= 1
            
            if num_good_neighbors < k+1: continue
            vertex_pairs.append((u,v))
    
    print("New vertex pairs:", vertex_pairs)

    for u,v in vertex_pairs:
        G_i.add_edge(u,v)
    
    return G_i


# Returns a treewidth and a tree decomposition using networkx
def treewidth(G: UndirectedGraph) -> Tuple[int, TreeDecomposition]:
    # uses heuristics but run it 20 times and get the best result
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

    # check if |E|
    num_e, num_v = len(G.edge_list), G.size
    # print("edge num:", num_e)
    # print("vertex num:", num_v)

    # note: this number is huge, small graphs will only hit one branch of the control flow
    c1 = 4*k**2 + 12*k + 16 # just a constant
    c2 = 1 / (8*k**2 + 24 *k + 32)
    d = 2*k**3 * (k+1) * c1

    # first condition because it is a default upperbound ( and will spoil the second condition if not added)
    # the second condition comes from Lemma 2.3
    if k <= num_v-1 and num_e > k * num_v - (k * (k+1))//2:
        print(num_e, k, num_v)
        print(k * num_v - (k * (k+1))//2)
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
        
        G_prime, new_edges = G.contract_graph(matching)
        new_edge_mapping = {v:u for u,v in new_edges.items()}
        contracted_edge_mapping = {v: u for u,v in matching}

        # yields a tree decomposition of G_prime of k width
        result = decompose(G_prime, k) # returns a TreeDecomposition Object
        
        # Lemma 3.4
        if result is False: 
            return False
        
        # Lemma 3.3 to reconstruct a graph with < 2k+1  width
        # print("G_prime tree width:", result.get_width())
        treedec = TreeDecomposition(G)
        for v, bags in result.vertex_bags.items():
            original_v = new_edge_mapping[v]
            for b in bags:
                treedec.add_to_bag(original_v, b)
        
        for retained_v, contracted_v in contracted_edge_mapping.items():
            # add contracted vertex to all bags with the retained vertex 
            for bag in treedec.vertex_bags[retained_v]:
                treedec.add_to_bag(contracted_v, bag)

        # print("G tree width:", treedec.get_width())
        
        # opted to neglect Theorem 2.4
        # TODO: given a tree decomposition of size l, determine if tree decomposition of size k exists
        
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
        
        

def test_simplicial_graph():
    g = UndirectedGraph(5)
    g.add_edge(1,2)
    g.add_edge(1,3)
    g.add_edge(1,4)
    g.add_edge(2,3)
    g.add_edge(3,4)
    g.add_edge(3,5)
    simplicial = sorted(g.get_simplicial_vertices())
    ans = [2,4,5]
    assert ans == simplicial, f"Failed simplicial test: {ans} != {simplicial}"

    g = UndirectedGraph(6)
    g.add_edge(1,2)
    g.add_edge(1,3)
    g.add_edge(1,4)
    g.add_edge(2,3)
    g.add_edge(2,5)
    g.add_edge(3,4)
    g.add_edge(3,5)
    g.add_edge(3,6)
    simplicial = sorted(g.get_simplicial_vertices())
    ans = [4,5,6]
    assert ans == simplicial, f"Failed simplicial test: {ans} != {simplicial}"


    print("Passed simplicial tests!")


def test_graph():
    random.seed(34)
    
    test_simplicial_graph()    

    g1 = generateRandomGraph(10,0.4)
    matching = g1.maximal_matching()
    contracted_graph = g1.contract_graph(matching)

    improved_graph = compute_improved_graph(g1, 4)
    print(improved_graph)


# test_networkx()
random.seed(32)
g1 = generateRandomGraph(50,0.8) # answer is 43
result = decompose(g1, 44)
if result is not False:
    print("Trewidth:", result.get_width())
else:
    print("Cannot get the k")

tw, td = treewidth(g1)
print(tw)
