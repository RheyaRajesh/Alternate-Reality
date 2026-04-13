import networkx as nx
import os
import httpx


def simulate_counterfactual(graph, removed_node, api_key):
    """
    Simulate what would happen if `removed_node` never occurred.

    1. Finds all descendants (effects) and ancestors (causes) of removed_node.
    2. Removes the node from a copy of the graph.
    3. Identifies effects that are now disconnected from root causes.
    4. Calls GPT-3.5-turbo for a narrative explanation.

    Returns a dict with:
        removed_node, disconnected_effects, ancestor_nodes,
        modified_graph, llm_explanation, summary
    """
    try:
        from openai import OpenAI

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

        # Find which effects are now truly lost (nodes that are NO LONGER reachable 
        # from ANY root cause in the modified graph)
        new_root_causes = [n for n in modified_graph.nodes() if modified_graph.in_degree(n) == 0]
        
        present_after = set()
        for r in new_root_causes:
            present_after.add(r)
            present_after.update(nx.descendants(modified_graph, r))
            
        disconnected_effects = list(descendants_before - present_after)

        # Build summary string
        summary = (
            f"If '{removed_node}' never happened: "
            f"{len(disconnected_effects)} downstream effect(s) would be prevented. "
            f"It was preceded by {len(ancestor_nodes)} cause(s)."
        )

        # Call GPT for explanation
        # Force-clear proxy environment variables to fix environment-specific library conflicts
        os.environ.pop("HTTP_PROXY", None)
        os.environ.pop("HTTPS_PROXY", None)
        os.environ.pop("http_proxy", None)
        os.environ.pop("https_proxy", None)
        
        from openai import OpenAI
        # Use Groq endpoint for fast, free inference
        client = OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
            http_client=httpx.Client()
        ) 
        gpt_response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert in causal reasoning and counterfactual analysis. "
                        "Give clear, logical explanations."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"In a causal chain, if '{removed_node}' never happened, what would be "
                        f"the alternate reality? The following effects would not occur: "
                        f"{disconnected_effects}. The following causes led to it: {ancestor_nodes}. "
                        f"Explain the alternate scenario in exactly 4 sentences. Be specific and logical."
                    ),
                },
            ],
            temperature=0.7,
            max_tokens=300,
        )

        llm_explanation = gpt_response.choices[0].message.content.strip()

        return {
            "removed_node": removed_node,
            "disconnected_effects": disconnected_effects,
            "ancestor_nodes": ancestor_nodes,
            "modified_graph": modified_graph,
            "llm_explanation": llm_explanation,
            "summary": summary,
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
