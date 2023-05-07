# CS270 Spring 2023 Final Project

This is a repository for a survey/implementation submitted as the final "project" for CS 270: Combinatorial Algorithms and Data Structures.

Tree-width is a crucial graph parameter, that has a significant role in domains like artificial intelligence, constraint-satisfaction and logical-circuit design. It is an especially interesting metric because most problems that are infeasible to compute on general graphs are solvable in linear time if a tree decomposition of bounded width is provided as input instead of the original unmodified graph. Determining if a graph $G$ has treewidth at most $k$ and producing the corresponding tree decomposition is known to be a NP-complete problem.

In this project, we implement the algorithm from [Bodlaender, '96](https://epubs.siam.org/doi/pdf/10.1137/S0097539793251219) which describes a linear-time algorithm for finding tree decompositions of small tree-width
