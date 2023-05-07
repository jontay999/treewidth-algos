import random
from typing import List, Tuple, Set, Dict
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
    
    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
    
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

