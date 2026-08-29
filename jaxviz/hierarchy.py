from collections import deque


def _is_cyclic(edges):
    outgoing = {}
    indegree = {}
    for source, target in edges:
        outgoing.setdefault(source, set())
        outgoing.setdefault(target, set())
        indegree.setdefault(source, 0)
        indegree.setdefault(target, 0)
        if target in outgoing[source]:
            continue
        outgoing[source].add(target)
        indegree[target] += 1

    ready = deque(
        node for node, degree in indegree.items()
        if degree == 0
    )
    visited = 0
    while ready:
        node = ready.popleft()
        visited += 1
        for target in outgoing[node]:
            indegree[target] -= 1
            if indegree[target] == 0:
                ready.append(target)
    return visited != len(indegree)


def _module_descendants(module, adjacency, ancestors):
    descendants = set()
    for node in adjacency:
        current = node
        while ancestors.get(current) is not None:
            current = ancestors[current]
            if current == module:
                descendants.add(node)
                break
    return descendants


def validate_collapsible_hierarchy(adjacency, ancestors, module_info):
    raw_edges = {
        (source, edge["target"])
        for source, data in adjacency.items()
        for edge in data["edges"]
    }
    if _is_cyclic(raw_edges):
        raise ValueError("The traced computation is cyclic")

    for module in module_info:
        descendants = _module_descendants(module, adjacency, ancestors)
        quotient_edges = set()
        for source, target in raw_edges:
            collapsed_source = module if source in descendants else source
            collapsed_target = module if target in descendants else target
            if collapsed_source != collapsed_target:
                quotient_edges.add((collapsed_source, collapsed_target))
        if not _is_cyclic(quotient_edges):
            continue
        module_name = module_info[module].get(
            "name", module_info[module].get("type", module)
        )
        raise ValueError(
            f"Module invocation {module_name!r} is not graph-convex"
        )
