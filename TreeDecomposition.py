from UndirectedGraph import UndirectedGraph

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

