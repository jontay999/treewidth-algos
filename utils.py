import random
from UndirectedGraph import UndirectedGraph
from TreeDecomposition import TreeDecomposition
from typing import Set, Callable

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

    for u,v in vertex_pairs:
        G_i.add_edge(u,v)
    
    return G_i


def generateRandomGraph(vertices: int, probability: float) -> UndirectedGraph:
    assert 0 < probability <= 1, "Probability has to be within 0-1"
    n = vertices
    graph = UndirectedGraph(n)

    # generate all edges
    all_edges = []
    for i in range(1,n+1):
        for j in range(i+1,n+1):
            if i == j: continue
            all_edges.append((i,j))

    for u,v in all_edges:
        if random.random() < probability:
            graph.add_edge(u,v)
    
    return graph

# check implementation of TreeDecomp class
def test_tree_decomposition_implementation():
    # using this graph: https://en.wikipedia.org/wiki/File:Tree_decomposition.svg
    g = UndirectedGraph(8)
    g.add_edge(1,2)
    g.add_edge(1,3)
    g.add_edge(2,3) 
    g.add_edge(2,5)
    g.add_edge(2,6)
    g.add_edge(2,7)
    g.add_edge(3,4)
    g.add_edge(3,5)
    g.add_edge(4,5)
    g.add_edge(5,7)
    g.add_edge(5,8)
    g.add_edge(6,7)
    g.add_edge(7,8)

    td = TreeDecomposition(g)
    bags_decomp = [
        [1,2,3],
        [2,3,5],
        [3,4,5],
        [2,5,7],
        [2,6,7],
        [5,7,8]
    ]
    for vertices in bags_decomp:
        bag = td.next_bag
        for v in vertices:
            td.add_to_bag(v, bag)
        td.increase_bag()

    return td


def test_treewidth(treewidth_solver: Callable, is_exact: bool = True, approx_ratio: int = None):
    
    # the approx ratio has to be given if its not exact
    if(not is_exact): assert approx_ratio

    # in the format (edge_list, vertices, answer)
    tests = [
        ([(1, 2), (1, 3), (1, 4), (3, 5), (4, 6)], 6, 1),
        ([(1, 2), (1, 3), (2, 3), (2, 5), (2, 6), (2, 7), (3, 4), (3, 5), (4, 5), (5, 7), (5, 8), (6, 7), (7, 8)], 8, 2),
        ([(1, 2), (1, 4), (2, 3), (2, 5), (3, 6), (4, 5), (4, 7), (5, 6), (5, 8), (6, 9), (7, 8), (8, 9)],9,3),
        ([(1, 2), (1, 3), (1, 4), (1, 5), (2, 3), (2, 4), (2, 5), (3, 4), (3, 5), (4, 5)],5,4),
        
        # series parallel graph -> treewidth <= 2
        ([(1,2),(2,3),(2,4),(2,5),(3,6),(4,6),(5,6),(6,8),(1,7),(7,8)], 8 , 2),

        # complete graph of 5 edges -> treewidth 4
        ([(1, 2),(1, 3),(1, 4),(1, 5),(2, 3),(2, 4),(2, 5),(3, 4),(3, 5),(4, 5)], 5, 4),

        # long test case 
        ([(1, 13),(1, 17),(1, 19),(1, 6),(2, 15),(2, 18),(2, 19),(2, 20),(13, 16),(13, 20),(13, 21),(15, 17),(15, 21),(15, 3),(16, 18),(16, 3),(16, 4),(17, 4),(17, 5),(18, 5),(18, 6),(19, 8),(19, 14),(20, 7),(20, 9),(21, 8),(21, 10),(3, 9),(3, 11),(4, 10),(4, 12),(5, 11),(5, 14),(6, 7),(6, 12),(7, 10),(7, 11),(8, 11),(8, 12),(9, 12),(9, 14),(10, 14)], 21, 8)
    ]

    correct = 0
    wrong = []
    for idx,(test, n, answer) in  enumerate(tests):
        graph = UndirectedGraph(n)
        for x,y in test: graph.add_edge(x,y)
        
        ans = None
        # call treewidth solver
        ans = treewidth_solver(graph, answer).width
        if (is_exact and ans == answer) or (not is_exact and ans <= approx_ratio * answer):
            correct += 1
        else:
            wrong.append(idx)
    
    
    kind_string = "exact" if is_exact else f"{approx_ratio}-approximation"
    print(f"Treewidth estimation ({kind_string})")
    print(f"Passed {correct}/{len(tests)} test cases")
    if len(wrong):
        for i in wrong:
            print(f"Failed test case {i}: {tests[i][0]}" )

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

