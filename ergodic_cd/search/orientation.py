from itertools import combinations, permutations
import pandas
import numpy as np
import networkx as nx


def orient_v_structures(mat: pandas.DataFrame, sepsets):
    """
    Orient X *--* Z *--* Y as X *--> Z <--* if X and Y are disjoint and Z is not in their separation set
    :param mat: the adjacency matrix
    :param sepsets: Separating sets, as a dict of tuples of nodes as keys and sets of nodes as values
    :return: the oriented matrix
    """

    nodes = mat.index

    results = []

    mat = mat.copy()

    def get_sepset(node_x, node_y):
        return sepsets.get((node_x, node_y), sepsets.get((node_y, node_x), set()))

    # check each node if it can serve as a collider for a disjoint neighbors
    for node_z in nodes:
        # check neighbors
        xy_nodes = set(mat.index[mat.loc[:, node_z] == 1])  # potential drivers of Z

        for node_x, node_y in combinations(xy_nodes, 2):

            if mat.loc[node_x, node_y] == 1:
                continue  # skip this pair as they are connected
            if node_z not in get_sepset(node_x, node_y):
                mat.loc[node_x, node_z] = 1
                mat.loc[node_z, node_x] = 0
                mat.loc[node_y, node_z] = 1
                mat.loc[node_z, node_y] = 0

                d = {}
                d["from_1"] = node_x
                d["from_2"] = node_y
                d["to"] = node_z
                results.append(d)

    return mat, results


def rule_1(mat: pandas.DataFrame):
    """
    [R1] Orient Z --> X --- Y into Z --> X --> Y if Z and Y are not connected.
    :param mat: the adjacency matrix
    :return: the oriented matrix and a list of results (orientations made)
    """
    nodes = mat.index
    results = []
    mat = mat.copy()

    def potential_parents_of(node):
        return set(mat.index[mat.loc[:, node] == 1])

    def potential_children_of(node):
        return set(mat.index[mat.loc[node] == 1])

    def direct_parents_of(node):
        return potential_parents_of(node) - potential_children_of(node)

    def undirected_neighbors(node):
        return potential_parents_of(node) & potential_children_of(node)

    def is_connected(node_a, node_b):
        return mat.loc[node_a, node_b] == 1 or mat.loc[node_b, node_a] == 1

    for node_y in nodes:
        x_nodes = undirected_neighbors(node_y)

        for node_x in x_nodes:
            z_nodes = direct_parents_of(node_x)

            for node_z in z_nodes:
                if not is_connected(node_z, node_y):
                    # Orient X --> Y
                    mat.loc[node_x, node_y] = 1
                    mat.loc[node_y, node_x] = 0

                    d = {"node_z": node_z, "node_x": node_x, "node_y": node_y}
                    results.append(d)
                    break  # Found an orientation for X --- Y, move to next X

    return mat, results


def rule_2(mat: pandas.DataFrame):
    """
    [R2] Orient X --- Y into X --> Y if there is a directed path X --> Z --> Y
    :param mat: the adjacency matrix
    :return: the oriented matrix and a list of results (orientations made)
    """
    nodes = mat.index
    results = []
    mat = mat.copy()

    def potential_parents_of(node):
        return set(mat.index[mat.loc[:, node] == 1])

    def potential_children_of(node):
        return set(mat.index[mat.loc[node] == 1])

    def direct_parents_of(node):
        return potential_parents_of(node) - potential_children_of(node)

    def undirected_neighbors(node):
        return potential_parents_of(node) & potential_children_of(node)

    for node_y in nodes:
        x_nodes = undirected_neighbors(node_y)
        z_nodes = direct_parents_of(node_y)

        for node_x in x_nodes:
            for node_z in z_nodes:
                if node_x in direct_parents_of(node_z):
                    # Orient X --> Y
                    mat.loc[node_x, node_y] = 1
                    mat.loc[node_y, node_x] = 0

                    d = {"node_x": node_x, "node_z": node_z, "node_y": node_y}
                    results.append(d)
                    break  # Found an orientation for X --- Y, move to next X

    return mat, results


def rule_3(mat: pandas.DataFrame):
    """
    [R3] Orient X --- Y into X --> Y if there exists X --- W --> Y and X --- Z --> Y,
    where W and Z are disconnected
    :param mat: the adjacency matrix
    :return: the oriented matrix and a list of results (orientations made)
    """
    from itertools import combinations

    nodes = mat.index
    results = []
    mat = mat.copy()

    def potential_parents_of(node):
        return set(mat.index[mat.loc[:, node] == 1])

    def potential_children_of(node):
        return set(mat.index[mat.loc[node] == 1])

    def direct_parents_of(node):
        return potential_parents_of(node) - potential_children_of(node)

    def undirected_neighbors(node):
        return potential_parents_of(node) & potential_children_of(node)

    def is_connected(node_a, node_b):
        return mat.loc[node_a, node_b] == 1 or mat.loc[node_b, node_a] == 1

    for node_y in nodes:
        x_nodes = undirected_neighbors(node_y)
        wz_nodes = direct_parents_of(node_y)

        for node_x in x_nodes:
            wz_nodes_of_x = undirected_neighbors(node_x).intersection(wz_nodes)

            for node_w, node_z in combinations(wz_nodes_of_x, 2):
                if is_connected(node_w, node_z):
                    continue  # W and Z are connected, skip

                # Orient X --> Y
                mat.loc[node_x, node_y] = 1
                mat.loc[node_y, node_x] = 0

                d = {
                    "node_x": node_x,
                    "node_y": node_y,
                    "node_w": node_w,
                    "node_z": node_z,
                }
                results.append(d)
                break  # Found an orientation for X --- Y, move to next X

    return mat, results


def rule_4(mat: pandas.DataFrame):
    """
    [R4] Orient X --- Y into X --> Y if W --> Z --> Y and X and Z are connected by an undirected edge,
    and W and Y are disconnected.
    :param mat: the adjacency matrix
    :return: the oriented matrix and a list of results (orientations made)
    """
    nodes = mat.index
    results = []
    mat = mat.copy()

    def potential_parents_of(node):
        return set(mat.index[mat.loc[:, node] == 1])

    def potential_children_of(node):
        return set(mat.index[mat.loc[node] == 1])

    def direct_parents_of(node):
        return potential_parents_of(node) - potential_children_of(node)

    def undirected_neighbors(node):
        return potential_parents_of(node) & potential_children_of(node)

    def is_connected(node_a, node_b):
        return mat.loc[node_a, node_b] == 1 or mat.loc[node_b, node_a] == 1

    for node_y in nodes:
        x_nodes = undirected_neighbors(node_y)
        z_nodes = direct_parents_of(node_y)

        for node_x in x_nodes:
            for node_z in z_nodes:
                if node_z not in undirected_neighbors(node_x):
                    continue  # Z and X are not connected by an undirected edge

                for node_w in direct_parents_of(node_z):
                    if is_connected(node_w, node_y):
                        continue  # W and Y are connected, skip

                    if node_w in undirected_neighbors(node_x):
                        # Orient X --> Y
                        mat.loc[node_x, node_y] = 1
                        mat.loc[node_y, node_x] = 0

                        d = {
                            "node_x": node_x,
                            "node_y": node_y,
                            "node_z": node_z,
                            "node_w": node_w,
                        }
                        results.append(d)
                        break  # Found an orientation for X --- Y, move to next Z

                if mat.loc[node_y, node_x] == 0:  # If we've oriented X --> Y
                    break  # Move to the next X

    return mat, results


def is_dag(matrix):
    """Check if the given adjacency matrix has cycles"""
    # first I will remove all undirected edges from the matrix
    matrix = matrix.copy()

    G = nx.DiGraph((matrix * (1 - matrix.T)))
    return nx.is_directed_acyclic_graph(G)


def generate_orientations(mat: pandas.DataFrame, unoriented_edges: list):
    if not unoriented_edges:
        yield mat
        return

    edge = unoriented_edges.pop()
    u, v = edge

    # Two possible orientations
    matrix1 = mat.copy()
    matrix1.loc[u, v] = 0

    matrix2 = mat.copy()
    matrix2.loc[v, u] = 0

    # Check if adding the edge creates a DAG
    if is_dag(matrix1):
        yield from generate_orientations(matrix1, unoriented_edges.copy())

    if is_dag(matrix2):
        yield from generate_orientations(matrix2, unoriented_edges.copy())


def get_unoriented_edges(mat: pandas.DataFrame):
    """gets the edges for which we have a forward and backward path"""

    edges = []
    for i in range(mat.shape[0]):
        for j in range(i + 1, mat.shape[1]):
            iname, jname = mat.index[i], mat.index[j]
            if mat.loc[iname, jname] == 1 and mat.loc[jname, iname] == 1:
                edges.append((iname, jname))
    return edges


def all_dags(init_matrix: pandas.DataFrame):
    """Main function to generate all DAGs from symmetric adjacency matrix"""
    edges = get_unoriented_edges(init_matrix)
    init_matrix = init_matrix.copy()

    dags = list(generate_orientations(init_matrix, edges))
    return dags


def generate_one_dag(init_matrix: pandas.DataFrame):
    """Main function to generate all DAGs from symmetric adjacency matrix"""
    edges = get_unoriented_edges(init_matrix)
    init_matrix = init_matrix.copy()

    try:
        return next(generate_orientations(init_matrix, edges))
    except:
        return None


def orient_by_rules(mat: pandas.DataFrame, rules: list):
    """
    Apply a sequence of orientation rules to a graph represented by an adjacency matrix.

    :param mat: The adjacency matrix
    :param rules: List of rule functions to apply (e.g., [rule_1, rule_2, rule_3, rule_4])
    :return: The oriented matrix and a dictionary with history of results for each rule
    """
    history = {}
    mat = mat.copy()

    rules_array = [None, rule_1, rule_2, rule_3, rule_4]

    # Apply each rule in sequence
    for i, rule_idx in enumerate(rules):
        rule = rules_array[rule_idx]
        if rule is None:
            continue

        mat, results = rule(mat)

        # Store the results in history
        rule_name = rule.__name__
        history[rule_name] = {"results": results, "count": len(results)}

    return mat, history
