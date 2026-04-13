import networkx as nx


def simulate_counterfactual(graph, removed_node, api_key):
    """
    Simulate what would happen if `removed_node` never occurred.

    1. Finds all descendants (effects) and ancestors (causes) of removed_node.
    2. Removes the node from a copy of the graph.
    3. Identifies effects that are no longer reachable from original roots.
    4. Builds a deterministic, graph-based explanation.

    Returns a dict with:
        removed_node, disconnected_effects, ancestor_nodes,
        modified_graph, llm_explanation, summary
    """
    try:
        if removed_node not in graph.nodes():
            return {
                "removed_node": removed_node,
                "disconnected_effects": [],
                "ancestor_nodes": [],
                "modified_graph": graph.copy(),
                "llm_explanation": "Node not found in graph.",
                "summary": "Node not found.",
            }

        # Descendants = nodes reachable FROM removed_node (effects)
        descendants_before = set(nx.descendants(graph, removed_node))

        # Ancestors = nodes that reach removed_node (causes)
        ancestor_nodes = list(nx.ancestors(graph, removed_node))

        # Create a modified copy without the removed node
        modified_graph = graph.copy()
        modified_graph.remove_node(removed_node)

        # Keep original roots (excluding removed node). We only count an effect
        # as "still happening" if it remains reachable from one of these roots.
        original_roots = [
            n for n in graph.nodes()
            if graph.in_degree(n) == 0 and n != removed_node
        ]

        disconnected_effects = []
        for node in descendants_before:
            still_reachable = any(
                nx.has_path(modified_graph, root, node)
                for root in original_roots
                if root in modified_graph
            )
            if not still_reachable:
                disconnected_effects.append(node)

        disconnected_effects = sorted(disconnected_effects)
        ancestor_nodes = sorted(ancestor_nodes)

        # Build summary string
        summary = (
            f"If '{removed_node}' never happened: "
            f"{len(disconnected_effects)} downstream effect(s) would be prevented. "
            f"It was preceded by {len(ancestor_nodes)} cause(s)."
        )

        # Deterministic explanation, strictly based on graph structure.
        if disconnected_effects:
            effects_text = ", ".join(disconnected_effects)
            llm_explanation = (
                f"Removing '{removed_node}' breaks causal reachability to: {effects_text}. "
                "These events become unreachable from the original root causes in the graph."
            )
        else:
            llm_explanation = (
                f"Removing '{removed_node}' does not disconnect any downstream nodes from "
                "the remaining original root causes in the graph."
            )

        if ancestor_nodes:
            llm_explanation += (
                f" Upstream causes of '{removed_node}' in the original graph are: "
                f"{', '.join(ancestor_nodes)}."
            )
        else:
            llm_explanation += (
                f" '{removed_node}' is a root cause in the original graph (no ancestors)."
            )

        return {
            "removed_node": removed_node,
            "disconnected_effects": disconnected_effects,
            "ancestor_nodes": ancestor_nodes,
            "modified_graph": modified_graph,
            "llm_explanation": llm_explanation,
            "summary": summary,
            "used_original_roots": original_roots,
        }

    except Exception as e:
        error_msg = str(e)
        error_lower = error_msg.lower()

        if "invalid api key" in error_lower or "authentication" in error_lower:
            friendly = "Invalid API key. Please check your key."
        elif "rate limit" in error_lower:
            friendly = "API rate limit reached. Please wait 60 seconds."
        else:
            friendly = f"Simulation error: {error_msg}"

        return {
            "removed_node": removed_node,
            "disconnected_effects": [],
            "ancestor_nodes": [],
            "modified_graph": graph.copy() if graph else None,
            "llm_explanation": friendly,
            "summary": f"Error during simulation: {friendly}",
            "error": True,
        }


def get_all_causal_paths(graph):
    """
    Find all simple paths between every pair of root causes and final effects.
    Returns a list of paths (each path is a list of node names).
    Limited to 10 paths for performance.
    """
    try:
        if graph is None or graph.number_of_nodes() == 0:
            return []

        root_causes = [n for n in graph.nodes() if graph.in_degree(n) == 0]
        final_effects = [n for n in graph.nodes() if graph.out_degree(n) == 0]

        all_paths = []
        for src in root_causes:
            for tgt in final_effects:
                if src == tgt:
                    continue
                try:
                    for path in nx.all_simple_paths(graph, src, tgt):
                        all_paths.append(path)
                        if len(all_paths) >= 10:
                            return all_paths
                except nx.NetworkXNoPath:
                    continue
                except Exception:
                    continue

        return all_paths

    except Exception as e:
        print(f"[Paths] Error finding paths: {e}")
        return []
