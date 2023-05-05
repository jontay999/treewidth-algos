import random
from typing import List, Tuple, Set, Dict, Union
import networkx as nx

class UndirectedGraph:
    def __init__(self, size: int):
        # assume graph is 1-indexed  vertices numbered 1...size
        assert size >= 1, "You need a graph of with at least 1 vertex"
        self.size = size
        self.vertices = set(range(1,size+1))
        self.edges = {i: set() for i in range(1,size+1)}
        self.edge_list = set()

    def add_edge(self, vertex1: int, vertex2: int):
        assert vertex1 in self.vertices and vertex2 in self.vertices, f"Valid vertices are only: {self.vertices}"
        if(vertex1 == vertex2): return # we don't ban self-loops but will not be taken into account
        self.edges[vertex1].add(vertex2)
        self.edges[vertex2].add(vertex1)
        if(vertex1 > vertex2):
            vertex1, vertex2 = vertex2, vertex1
        self.edge_list.add((vertex1, vertex2))
    
    def __str__(self) -> str:
        string_rep = "Undirected graph with {} vertices\n".format(self.size)
        for i in self.vertices:
            curr_edges = sorted(list(self.edges[i]))
            string_rep += str(i) + ": "
            for edge in curr_edges:
                string_rep += str(edge) + ", "
            string_rep += "\n"
        return string_rep
    
    def vertex_degrees(self) -> Dict[int, int]:
        deg = {}
        for v in self.vertices:
            deg[v] = len(self.edges[v])
        return deg
    
    # 2-approximation greedy algorithm
    def maximal_matching(self) -> Set[Tuple[int, int]]:
        seen = set()
        matching = set()
        for u,v in self.edge_list:
            if(u not in seen and v not in seen):
                matching.add((u,v))
                seen.add(u)
                seen.add(v)
        return matching
    
    # contract graph given a matching
    def contract_graph(self, matching: Set[Tuple[int, int]]) -> Tuple["UndirectedGraph", Dict[int,int]]:
        # u merges with v to become a big node
        mapping = {u: v for u,v in matching}
        new_size = self.size - len(matching)
        g = UndirectedGraph(new_size)

        # number new edges appropriately
        new_edges = {}
        curr_num = 1
        for i in self.vertices:
            if(i not in mapping):
                new_edges[i] = curr_num
                curr_num += 1

        # create new edges
        for node, neighbors in self.edges.items():
            # get new index, accounts for if node has been contracted 
            if node in mapping:
                node1 = new_edges[mapping[node]]
            else:
                node1 = new_edges[node]
            

            for neigh in neighbors:
                # contracted edge does not appear in new graph
                if (node, neigh) in matching:
                    continue

                if neigh in mapping:
                    node2 = new_edges[mapping[neigh]]
                else:
                    node2 = new_edges[neigh]
                
                g.add_edge(node1, node2)
        
        return g, new_edges
    
    # definition: neighbors form a clique
    def get_simplicial_vertices(self) -> List[int]:
        # TO IMPROVE: currently O(V^3) -> can use a queue and 2 bucket sorts to make it linear
        simplicial_vertices = []
        for u in self.vertices:
            neighbors = self.edges[u] 
            if all(v in self.edges[w] for v in neighbors for w in neighbors if v != w):
                simplicial_vertices.append(u)
        return simplicial_vertices
    
    # checks if graph is simplicial
    def is_simplicial(self) -> bool:
        return len(self.get_simplicial_vertices()) == self.size
    


    def subgraph(self, nodes: Set[int]) -> "UndirectedGraph":
        num_v = len(nodes)
        sub_g = UndirectedGraph(num_v)
        mapping = {}
        curr = 1
        for node in nodes:
            mapping[node] = curr
            curr += 1
        
        for node in nodes:
            for neighbor in self.edges[node]:
                if neighbor not in nodes: continue
                u,v = mapping[node], mapping[neighbor]
                sub_g.add_edge(u,v)
        return sub_g

    # create deep copy of current graph
    def copy(self) -> "UndirectedGraph":
        new_graph = UndirectedGraph(self.size)
        for u,v in self.edge_list:
            new_graph.add_edge(u,v)
        return new_graph
    
    # will throw an error if edge does not exist
    def remove_edge(self, u: int, v:int):
        assert u in self.vertices and v in self.vertices, f"{u} or {v} are not valid vertices"
        assert u in self.edges[v] and v in self.edges[u], f"{u}-{v} is not a valid edge"
        if u > v: u,v = v,u
        assert (u,v) in self.edge_list, f"Something probably went wrong if it only threw an error here"
        self.edges[u].remove(v)
        self.edges[v].remove(u)
        self.edge_list.remove((u,v))


    def remove_node(self, node: int):
        assert node in self.vertices, "Not valid vertex"
        for neighbor in self.edges[node]:
            if node > neighbor:
                u,v = neighbor, node
            else:
                u,v = node, neighbor
            
            # only need to remove the neighbor, as we will completely delete the self.edges[node] after
            self.edges[neighbor].remove(node)
            self.edge_list.remove((u,v))
        
        del self.edges[node]
        self.vertices.remove(node)
        self.size -= 1

    def add_node(self, node: int):
        assert node not in self.vertices
        self.size += 1
        self.vertices.add(node)


    def convert_to_nx(self) -> nx.Graph:
        nx_graph = nx.Graph()
        for u,v in self.edge_list:
            nx_graph.add_edge(u,v)
        return nx_graph
    
    # outputs graph in a .gr format
    def write_to_file(self, filename: str):
        with open(filename, 'w') as f:
            f.write(f"p tw {len(self.vertices)} {len(self.edge_list)}\n")
            for u,v in self.edge_list:
                f.write(f"{u} {v}\n")
        


    # returns a graph with same structure with renumbered vertices
    def randomize(self) -> Tuple["UndirectedGraph", Dict[int,int]]:
        new_graph = UndirectedGraph(self.size)
        new_vertices = list(self.vertices)
        original_mapping = {v: i for i, v in enumerate(new_vertices)}

        random.shuffle(new_vertices)
        new_mapping = {i:v for i,v in enumerate(new_vertices)}

        restored_mapping = {}
        for u,v in self.edge_list:
            new_u = new_mapping[original_mapping[u]]
            new_v = new_mapping[original_mapping[v]]
            restored_mapping[new_u] = u
            restored_mapping[new_v] = v
            new_graph.add_edge(new_u, new_v)
        return new_graph, restored_mapping

        

class TreeDecomposition():
    def __init__(self, graph: UndirectedGraph):
        self.graph = graph

        # to figure out what to call the next bag
        self.next_bag = 1

        # vertex -> bags
        self.vertex_bags = {i: set() for i in self.graph.vertices}

        # bags -> vertex
        self.bags = {}

        self.width = 0

    def add_to_bag(self, vertex: int, bag: int):
        assert vertex in self.graph.vertices, f"Vertices have to be in {self.graph.vertices}"
        if(bag not in self.bags): self.bags[bag] = set()
        self.vertex_bags[vertex].add(bag)
        self.bags[bag].add(vertex)
        if len(self.bags[bag]) - 1 > self.width:
            self.width = len(self.bags[bag]) - 1

    def get_width(self) -> int:
        return self.width
    
    def increase_bag(self):
        self.next_bag += 1

    def __str__(self) -> str:
        string = ""

        string += "Bags:\n"
        for bag, vertices in self.bags.items():
            string += f"Bag {bag}: {vertices}\n"
        
        return string




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



def test_instances(is_exact = True, approx_ratio = None):
    
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
        # ans = treewidth(graph)
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
    



def known_graph():
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




if __name__ == "__main__":
    g1 = generateRandomGraph(5,0.6)
    g2 = generateRandomGraph(8,0.5)
    td = known_graph()
    print(td)
    print(td.get_width())
