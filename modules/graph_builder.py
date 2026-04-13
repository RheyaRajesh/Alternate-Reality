"""
Module 3: Graph Building & Visualization
Builds directed causal graphs using NetworkX and visualizes them with Matplotlib.
"""

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def build_causal_graph(causal_pairs):
    """
    Build a directed causal graph from a list of causal pairs.
    Each pair: {"cause": ..., "effect": ..., "confidence": ...}
    Returns a networkx DiGraph.
    """
    graph = nx.DiGraph()

    for pair in causal_pairs:
        cause = pair.get("cause", "").strip()
        effect = pair.get("effect", "").strip()
        confidence = float(pair.get("confidence", 0.5))

        if cause and effect:
            graph.add_edge(cause, effect, confidence=confidence)

    return graph


def get_node_colors(graph):
    """
    Determine colors for each node based on its role in the causal chain.
    - Red (#ff6b6b): Pure cause (only outgoing edges)
    - Cyan (#4ecdc4): Pure effect (only incoming edges)
    - Purple (#a855f7): Intermediate (both incoming and outgoing)
    Returns a list of color strings in the same order as graph.nodes().
    """
    colors = []
    for node in graph.nodes():
        in_degree = graph.in_degree(node)
        out_degree = graph.out_degree(node)

        if in_degree == 0 and out_degree > 0:
            colors.append("#ff6b6b")   # Root cause — red
        elif in_degree > 0 and out_degree == 0:
            colors.append("#4ecdc4")   # Final effect — cyan
        else:
            colors.append("#a855f7")   # Intermediate — purple

    return colors


def visualize_graph(graph, title="Causal Graph", figsize=(12, 8)):
    """
    Create a matplotlib figure visualizing the causal graph with dark theme styling.
    Returns the matplotlib Figure object.
    """
    fig = plt.figure(figsize=figsize)
    fig.patch.set_facecolor("#0d1117")

    ax = fig.add_subplot(111)
    ax.set_facecolor("#0d1117")

    if graph is None or graph.number_of_nodes() == 0:
        ax.text(
            0.5, 0.5,
            "No causal relationships found",
            ha="center", va="center",
            color="white", fontsize=16,
            transform=ax.transAxes
        )
        ax.axis("off")
        ax.set_title(title, color="white", fontsize=14, pad=15)
        return fig

    # Layout
    pos = nx.spring_layout(graph, seed=42, k=2)

    node_colors = get_node_colors(graph)

    # Draw nodes
    nx.draw_networkx_nodes(
        graph, pos,
        node_color=node_colors,
        node_size=2000,
        alpha=0.9,
        ax=ax
    )

    # Draw edges
    nx.draw_networkx_edges(
        graph, pos,
        edge_color="#00bcd4",
        arrows=True,
        arrowsize=20,
        width=2,
        connectionstyle="arc3,rad=0.1",
        ax=ax
    )

    # Draw node labels
    nx.draw_networkx_labels(
        graph, pos,
        font_color="white",
        font_size=9,
        font_weight="bold",
        ax=ax
    )

    # Draw edge confidence labels
    edge_labels = {
        (u, v): f"{d.get('confidence', 0):.2f}"
        for u, v, d in graph.edges(data=True)
    }
    nx.draw_networkx_edge_labels(
        graph, pos,
        edge_labels=edge_labels,
        font_color="#ffd700",
        font_size=7,
        ax=ax
    )

    # Title
    ax.set_title(title, color="white", fontsize=14, pad=15, fontweight="bold")

    # Legend
    legend_patches = [
        mpatches.Patch(color="#ff6b6b", label="Root Cause"),
        mpatches.Patch(color="#4ecdc4", label="Final Effect"),
        mpatches.Patch(color="#a855f7", label="Intermediate"),
    ]
    ax.legend(
        handles=legend_patches,
        loc="lower right",
        facecolor="#161b22",
        edgecolor="#30363d",
        labelcolor="white",
        fontsize=9
    )

    ax.axis("off")
    plt.tight_layout()

    return fig


def get_graph_stats(graph):
    """
    Return a dictionary of statistics about the causal graph.
    Keys: nodes, edges, root_causes, final_effects, longest_chain
    """
    if graph is None:
        return {
            "nodes": 0,
            "edges": 0,
            "root_causes": [],
            "final_effects": [],
            "longest_chain": 0,
        }

    root_causes = [n for n in graph.nodes() if graph.in_degree(n) == 0]
    final_effects = [n for n in graph.nodes() if graph.out_degree(n) == 0]

    longest_chain = 0
    try:
        if nx.is_directed_acyclic_graph(graph):
            longest_chain = nx.dag_longest_path_length(graph)
        else:
            # Fall back to a heuristic for graphs with cycles
            longest_chain = max(
                (len(path) - 1)
                for src in root_causes
                for tgt in final_effects
                for path in nx.all_simple_paths(graph, src, tgt)
            ) if root_causes and final_effects else 0
    except Exception:
        longest_chain = graph.number_of_edges()

    return {
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "root_causes": root_causes,
        "final_effects": final_effects,
        "longest_chain": longest_chain,
    }
