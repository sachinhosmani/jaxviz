"""Graph helpers shared by the renderer."""


def build_immediate_ancestor_map(ancestor_dict, adj_list):
    """Turn each node's full ancestor chain (immediate-parent-first) into a
    child -> immediate-parent map, the structure the frontend uses to build the
    collapsible module containers.
    """
    immediate_ancestor_map = {}
    for node, ancestors in ancestor_dict.items():
        if ancestors and node in adj_list:
            immediate_ancestor_map[node] = ancestors[0]
            for i in range(len(ancestors) - 1):
                if ancestors[i] not in immediate_ancestor_map:
                    immediate_ancestor_map[ancestors[i]] = ancestors[i + 1]
    return immediate_ancestor_map
