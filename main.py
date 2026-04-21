"""
Alternate Reality Engine — Main Streamlit Application
A Causal RAG System for Counterfactual Reasoning
"""

import streamlit as st
import json
import time

from modules.ingestion import initialize_chromadb, embed_and_store, retrieve_relevant_chunks
from modules.causal_extraction import extract_causal_pairs, format_causal_chain
from modules.graph_builder import build_causal_graph, visualize_graph, get_graph_stats
from modules.counterfactual import simulate_counterfactual, get_all_causal_paths

# ─────────────────────────────────────────────────────────────
# PAGE CONFIGURATION
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Alternate Reality Engine",
    page_icon="🌀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# CSS STYLING
# ─────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ── Global ── */
    html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
        background-color: #0d1117 !important;
        color: white !important;
        font-family: 'Inter', 'Segoe UI', sans-serif;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background-color: #161b22 !important;
        border-right: 1px solid #30363d;
    }
    [data-testid="stSidebar"] * { color: white !important; }

    /* ── Text inputs & textareas ── */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea,
    .stSelectbox > div > div {
        background-color: #21262d !important;
        color: white !important;
        border: 1px solid #00bcd4 !important;
        border-radius: 8px !important;
    }

    /* ── Buttons ── */
    .stButton > button {
        background-color: #00bcd4 !important;
        color: black !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 0.5rem 1.5rem !important;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        background-color: #0097a7 !important;
        transform: translateY(-1px);
        box-shadow: 0 4px 15px rgba(0, 188, 212, 0.4);
    }

    /* ── Cards / Containers ── */
    .are-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 1rem 1.5rem;
        margin-bottom: 1rem;
    }

    /* ── Chain display ── */
    .chain-box {
        background-color: #161b22;
        border: 1px solid #00bcd4;
        border-radius: 8px;
        padding: 1rem 1.5rem;
        font-size: 1.1rem;
        color: #00bcd4;
        font-weight: 600;
        margin: 0.5rem 0 1rem 0;
        word-break: break-word;
    }

    /* ── Success / Error / Warning boxes ── */
    .are-success {
        background-color: #1a4731;
        border: 1px solid #2ea043;
        border-radius: 8px;
        padding: 1rem;
        color: #56d364;
        font-weight: 600;
    }
    .are-error {
        background-color: #4d1a1a;
        border: 1px solid #f85149;
        border-radius: 8px;
        padding: 1rem;
        color: #f85149;
    }
    .are-warning {
        background-color: #3d2e00;
        border: 1px solid #d29922;
        border-radius: 8px;
        padding: 1rem;
        color: #e3b341;
    }

    /* ── Metric cards ── */
    .metric-card {
        background-color: #161b22;
        border-left: 4px solid #00bcd4;
        border-radius: 6px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.5rem;
    }
    .metric-card .metric-label {
        font-size: 0.75rem;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-card .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #00bcd4;
    }

    /* ── LLM explanation box ── */
    .llm-box {
        background-color: #161b22;
        border-left: 4px solid #00bcd4;
        border-radius: 6px;
        padding: 1.25rem 1.5rem;
        font-style: italic;
        font-size: 1.05rem;
        line-height: 1.7;
        color: #e6edf3;
    }

    /* ── Badge styles ── */
    .badge-red {
        display: inline-block;
        background-color: rgba(248, 81, 73, 0.15);
        border: 1px solid #f85149;
        color: #f85149;
        border-radius: 20px;
        padding: 0.2rem 0.75rem;
        font-size: 0.85rem;
        margin: 0.2rem;
    }
    .badge-yellow {
        display: inline-block;
        background-color: rgba(210, 153, 34, 0.15);
        border: 1px solid #d29922;
        color: #e3b341;
        border-radius: 20px;
        padding: 0.2rem 0.75rem;
        font-size: 0.85rem;
        margin: 0.2rem;
    }
    .badge-cyan {
        display: inline-block;
        background-color: rgba(0, 188, 212, 0.15);
        border: 1px solid #00bcd4;
        color: #00bcd4;
        border-radius: 20px;
        padding: 0.2rem 0.75rem;
        font-size: 0.85rem;
        margin: 0.2rem;
    }
    .badge-purple {
        display: inline-block;
        background-color: rgba(168, 85, 247, 0.15);
        border: 1px solid #a855f7;
        color: #c084fc;
        border-radius: 20px;
        padding: 0.2rem 0.75rem;
        font-size: 0.85rem;
        margin: 0.2rem;
    }

    /* ── Spinner text ── */
    .stSpinner > div > div { color: white !important; }

    /* ── Tab styling ── */
    [data-testid="stTabs"] button {
        color: #8b949e !important;
        font-weight: 500;
    }
    [data-testid="stTabs"] button[aria-selected="true"] {
        color: #00bcd4 !important;
        border-bottom: 2px solid #00bcd4 !important;
    }

    /* ── Expander ── */
    [data-testid="stExpander"] {
        background-color: #161b22 !important;
        border: 1px solid #30363d !important;
        border-radius: 8px !important;
    }

    /* ── File uploader ── */
    [data-testid="stFileUploadDropzone"] {
        background-color: #21262d !important;
        border: 2px dashed #00bcd4 !important;
        border-radius: 8px !important;
    }

    /* ── Selectbox dropdown ── */
    .stSelectbox [data-baseweb="select"] > div {
        background-color: #21262d !important;
        border-color: #00bcd4 !important;
    }

    /* ── Progress bar ── */
    .stProgress > div > div > div {
        background-color: #00bcd4 !important;
    }

    /* ── Footer ── */
    .are-footer {
        color: #8b949e;
        font-size: 0.8rem;
        text-align: center;
    }

    /* ── Divider ── */
    hr { border-color: #30363d !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────
# SESSION STATE INITIALIZATION
# ─────────────────────────────────────────────────────────────
defaults = {
    "collection": None,
    "causal_pairs": [],
    "graph": None,
    "chain_text": "",
    "query": "",
    "retrieved_chunks": [],
    "data_loaded": False,
    "analysis_done": False,
    "counterfactual_result": None,
    "api_key": "",
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<h1 style='color:#00bcd4; font-size:2.2rem; margin-bottom:0;'>🌀 ARE</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='color:white; font-size:1rem; margin-top:0;'>Alternate Reality Engine</p>",
        unsafe_allow_html=True,
    )
    st.divider()

    st.markdown("### ⚙️ Configuration")
    api_key_input = st.text_input(
        "Groq API Key",
        type="password",
        value=st.session_state.api_key,
        placeholder="gsk_...",
        key="api_key_input",
    )
    if api_key_input:
        st.session_state.api_key = api_key_input

    st.markdown(
        "<small style='color:#8b949e;'>Get a free API key at "
        "<a href='https://console.groq.com' target='_blank' "
        "style='color:#00bcd4;'>console.groq.com</a></small>",
        unsafe_allow_html=True,
    )
    st.divider()

    st.markdown("### 📊 Pipeline Status")
    status_items = [
        ("Step 1 · Data Loaded", st.session_state.data_loaded),
        ("Step 2 · Query Analyzed", st.session_state.analysis_done),
        ("Step 3 · Graph Built", st.session_state.graph is not None),
        (
            "Step 4 · Simulation Ready",
            st.session_state.graph is not None
            and st.session_state.graph.number_of_nodes() > 0,
        ),
    ]
    for label, done in status_items:
        icon = "✅" if done else "❌"
        color = "#56d364" if done else "#f85149"
        st.markdown(
            f"<div style='margin:0.3rem 0; color:{color};'>{icon} {label}</div>",
            unsafe_allow_html=True,
        )
    st.divider()

    with st.expander("📖 How to Use"):
        st.markdown(
            """
1. Enter your **OpenAI API key** above  
2. Upload a text file or use **sample data**  
3. Type your **question** in the query box  
4. Click **Analyze** to build the causal chain  
5. Select a node and **simulate** what-if scenarios  
            """
        )


# ─────────────────────────────────────────────────────────────
# MAIN HEADER
# ─────────────────────────────────────────────────────────────
st.markdown(
    "<h1 style='text-align:center; color:white; font-size:2.8rem; "
    "margin-bottom:0.2rem;'>🌀 Alternate Reality Engine</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align:center; color:#00bcd4; font-size:1.2rem; "
    "margin-bottom:0.3rem;'>Causal RAG for Counterfactual Reasoning</p>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align:center; color:#8b949e; font-size:0.95rem; "
    "margin-bottom:1rem;'>Upload knowledge, extract causal chains, and simulate "
    "alternate realities with AI.</p>",
    unsafe_allow_html=True,
)
st.divider()


# ─────────────────────────────────────────────────────────────
# TAB NAVIGATION
# ─────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📁 Step 1: Load Data",
        "🔍 Step 2: Analyze Query",
        "🕸️ Step 3: Causal Graph",
        "🔀 Step 4: Simulate Reality",
    ]
)


# ═══════════════════════════════════════════════════════════════
# TAB 1 — LOAD DATA
# ═══════════════════════════════════════════════════════════════
with tab1:
    st.markdown("## Upload Your Knowledge Base")

    # ── Left: Upload file ──
    st.subheader("📤 Upload Text File")
    uploaded_file = st.file_uploader(
        "Choose a .txt file",
        type=["txt"],
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        file_content = uploaded_file.read().decode("utf-8")

        with st.expander("📄 File Preview (first 500 chars)"):
            st.text(file_content[:500] + ("..." if len(file_content) > 500 else ""))

        if st.button("💾 Store in Knowledge Base", key="btn_store_upload"):
            try:
                with st.spinner("Initializing in-memory knowledge store..."):
                    collection = initialize_chromadb()
                    if collection is None:
                        st.markdown(
                            "<div class='are-error'>❌ Knowledge store error. Please reload the page.</div>",
                            unsafe_allow_html=True,
                        )
                        st.stop()
                    st.session_state.collection = collection

                with st.spinner("Processing and embedding text..."):
                    n_chunks = embed_and_store(file_content, collection, source_name="uploaded_doc")

                st.markdown(
                    f"<div class='are-success'>✅ Stored {n_chunks} chunk(s) in the knowledge base.</div>",
                    unsafe_allow_html=True,
                )
                st.session_state.data_loaded = True

            except Exception as e:
                st.markdown(
                    f"<div class='are-error'>❌ Error: {e}</div>",
                    unsafe_allow_html=True,
                )

    # ── Ready banner ──
    if st.session_state.data_loaded:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            "<div class='are-success' style='text-align:center; font-size:1.1rem;'>"
            "✅ Knowledge base is ready! Proceed to Step 2."
            "</div>",
            unsafe_allow_html=True,
        )


# ═══════════════════════════════════════════════════════════════
# TAB 2 — ANALYZE QUERY
# ═══════════════════════════════════════════════════════════════
with tab2:
    if not st.session_state.data_loaded:
        st.markdown(
            "<div class='are-warning'>⚠️ Please load data in Step 1 first.</div>",
            unsafe_allow_html=True,
        )

    if not st.session_state.api_key:
        st.markdown(
            "<div class='are-warning'>⚠️ Please enter your OpenAI API key in the sidebar.</div>",
            unsafe_allow_html=True,
        )

    st.markdown("## Ask a Causal Question")

    st.markdown(
        "<div class='are-card'>"
        "<b>💡 Example queries:</b><br>"
        "• <i>Why did the system crash?</i><br>"
        "• <i>What caused the climate change?</i><br>"
        "• <i>Why did revenue decrease?</i>"
        "</div>",
        unsafe_allow_html=True,
    )

    query_input = st.text_area(
        "Your question",
        value=st.session_state.query,
        placeholder="Why did the system crash?",
        height=100,
        label_visibility="collapsed",
        key="query_input",
    )
    if query_input:
        st.session_state.query = query_input

    if st.button("🔍 Analyze Causal Chain", type="primary", key="btn_analyze"):
        # Validation
        if not st.session_state.api_key:
            st.markdown(
                "<div class='are-error'>❌ Please enter your OpenAI API key in the sidebar.</div>",
                unsafe_allow_html=True,
            )
            st.stop()

        if not st.session_state.query.strip():
            st.markdown(
                "<div class='are-error'>❌ Please enter a query before analyzing.</div>",
                unsafe_allow_html=True,
            )
            st.stop()

        if st.session_state.collection is None:
            st.markdown(
                "<div class='are-error'>❌ Knowledge store not initialized. Please load data in Step 1.</div>",
                unsafe_allow_html=True,
            )
            st.stop()

        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            # ── Step 1: Retrieve chunks ──
            status_text.markdown(
                "<p style='color:#00bcd4;'>🔍 Searching knowledge base...</p>",
                unsafe_allow_html=True,
            )
            progress_bar.progress(5)

            chunks = retrieve_relevant_chunks(
                st.session_state.query,
                st.session_state.collection,
                top_k=3,
            )
            st.session_state.retrieved_chunks = chunks
            progress_bar.progress(25)
            status_text.markdown(
                f"<p style='color:#56d364;'>✅ Found {len(chunks)} relevant chunk(s).</p>",
                unsafe_allow_html=True,
            )

            if not chunks:
                st.markdown(
                    "<div class='are-warning'>⚠️ No relevant chunks found. "
                    "Try rephrasing your query or loading more data.</div>",
                    unsafe_allow_html=True,
                )

            # ── Step 2: Extract causal pairs ──
            status_text.markdown(
                "<p style='color:#00bcd4;'>🧠 Extracting causal relationships...</p>",
                unsafe_allow_html=True,
            )
            progress_bar.progress(30)

            pairs = extract_causal_pairs(chunks, st.session_state.api_key)
            st.session_state.causal_pairs = pairs
            progress_bar.progress(50)
            status_text.markdown(
                f"<p style='color:#56d364;'>✅ Extracted {len(pairs)} causal pair(s).</p>",
                unsafe_allow_html=True,
            )

            if not pairs:
                st.markdown(
                    "<div class='are-warning'>⚠️ Could not extract causal relationships. "
                    "Try rephrasing your query or adding more context.</div>",
                    unsafe_allow_html=True,
                )

            # ── Step 3: Build graph ──
            status_text.markdown(
                "<p style='color:#00bcd4;'>🕸️ Building causal graph...</p>",
                unsafe_allow_html=True,
            )
            progress_bar.progress(55)

            graph = build_causal_graph(pairs)
            st.session_state.graph = graph

            chain_text = format_causal_chain(pairs)
            st.session_state.chain_text = chain_text
            progress_bar.progress(75)

            # ── Step 4: Finalize ──
            status_text.markdown(
                "<p style='color:#00bcd4;'>✨ Finalizing analysis...</p>",
                unsafe_allow_html=True,
            )
            time.sleep(0.5)
            progress_bar.progress(100)
            st.session_state.analysis_done = True

            status_text.markdown(
                "<div class='are-success'>✅ Analysis complete! View results in Steps 3 and 4.</div>",
                unsafe_allow_html=True,
            )

        except ValueError as ve:
            # API key / rate limit errors
            progress_bar.empty()
            status_text.empty()
            st.markdown(
                f"<div class='are-error'>❌ {ve}</div>",
                unsafe_allow_html=True,
            )
        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            err = str(e).lower()
            if "invalid api key" in err or "authentication" in err:
                msg = "Invalid API key. Please check your key."
            elif "rate limit" in err:
                msg = "API rate limit reached. Please wait 60 seconds."
            else:
                msg = f"An error occurred: {e}"
            st.markdown(
                f"<div class='are-error'>❌ {msg}</div>",
                unsafe_allow_html=True,
            )

    # ── Results section ──
    if st.session_state.analysis_done and st.session_state.causal_pairs:
        st.markdown("---")
        st.markdown("## 📊 Analysis Results")

        m1, m2, m3, m4 = st.columns(4)
        metrics = [
            ("Chunks Retrieved", len(st.session_state.retrieved_chunks)),
            ("Causal Pairs", len(st.session_state.causal_pairs)),
            (
                "Graph Nodes",
                st.session_state.graph.number_of_nodes()
                if st.session_state.graph
                else 0,
            ),
            (
                "Graph Edges",
                st.session_state.graph.number_of_edges()
                if st.session_state.graph
                else 0,
            ),
        ]
        for col, (label, value) in zip([m1, m2, m3, m4], metrics):
            with col:
                st.markdown(
                    f"<div class='metric-card'>"
                    f"<div class='metric-label'>{label}</div>"
                    f"<div class='metric-value'>{value}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        # Chain display
        st.markdown("### 🔗 Causal Chain")
        st.markdown(
            f"<div class='chain-box'>{st.session_state.chain_text}</div>",
            unsafe_allow_html=True,
        )

        # Pairs list
        with st.expander("📋 Extracted Causal Pairs"):
            for i, pair in enumerate(st.session_state.causal_pairs):
                conf_pct = int(float(pair.get("confidence", 0)) * 100)
                st.markdown(
                    f"<div class='are-card' style='margin-bottom:0.5rem;'>"
                    f"<span class='badge-red'>{pair['cause']}</span>"
                    f" &nbsp;→&nbsp; "
                    f"<span class='badge-cyan'>{pair['effect']}</span>"
                    f"&nbsp;&nbsp;<span class='badge-yellow'>conf: {conf_pct}%</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

    # Retrieved chunks expander
    if st.session_state.retrieved_chunks:
        with st.expander("📄 Retrieved Context"):
            for idx, chunk in enumerate(st.session_state.retrieved_chunks):
                st.markdown(
                    f"<div class='are-card'>"
                    f"<b style='color:#00bcd4;'>Chunk {idx + 1}</b><br>"
                    f"<span style='color:#e6edf3;'>{chunk}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )


# ═══════════════════════════════════════════════════════════════
# TAB 3 — CAUSAL GRAPH
# ═══════════════════════════════════════════════════════════════
with tab3:
    if not st.session_state.analysis_done:
        st.markdown(
            "<div class='are-warning'>⚠️ Please complete Step 2 (Analyze Query) first.</div>",
            unsafe_allow_html=True,
        )

    elif st.session_state.graph is not None and st.session_state.graph.number_of_nodes() > 0:
        graph = st.session_state.graph
        col_graph, col_stats = st.columns([7, 3])

        # ── Left: Graph visualization ──
        with col_graph:
            st.subheader("🕸️ Causal Relationship Graph")
            try:
                fig = visualize_graph(graph, title="Causal Chain")
                st.pyplot(fig)
            except Exception as e:
                st.markdown(
                    f"<div class='are-error'>❌ Error rendering graph: {e}</div>",
                    unsafe_allow_html=True,
                )
            st.caption(
                "🔴 Root Causes  |  🟣 Intermediate Nodes  |  🩵 Final Effects  "
                "|  Edge labels show confidence scores"
            )

        # ── Right: Stats ──
        with col_stats:
            st.subheader("📊 Graph Statistics")
            stats = get_graph_stats(graph)

            stat_items = [
                ("Total Nodes", stats["nodes"]),
                ("Total Edges", stats["edges"]),
                ("Longest Chain", stats["longest_chain"]),
            ]
            for label, val in stat_items:
                st.markdown(
                    f"<div class='metric-card'>"
                    f"<div class='metric-label'>{label}</div>"
                    f"<div class='metric-value'>{val}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

            st.markdown("#### 🌱 Root Causes")
            if stats["root_causes"]:
                for rc in stats["root_causes"]:
                    st.markdown(
                        f"<span class='badge-red'>{rc}</span>",
                        unsafe_allow_html=True,
                    )
            else:
                st.markdown("<i style='color:#8b949e;'>None found</i>", unsafe_allow_html=True)

            st.markdown("#### 🎯 Final Effects")
            if stats["final_effects"]:
                for fe in stats["final_effects"]:
                    st.markdown(
                        f"<span class='badge-cyan'>{fe}</span>",
                        unsafe_allow_html=True,
                    )
            else:
                st.markdown("<i style='color:#8b949e;'>None found</i>", unsafe_allow_html=True)

        # ── Below: All paths ──
        st.markdown("---")
        st.subheader("🛤️ All Causal Paths")
        try:
            paths = get_all_causal_paths(graph)
            if paths:
                for i, path in enumerate(paths):
                    path_str = " → ".join(path)
                    st.markdown(
                        f"<div class='are-card' style='margin-bottom:0.4rem;'>"
                        f"<b style='color:#00bcd4;'>Path {i + 1}:</b> "
                        f"<span style='color:#e6edf3;'>{path_str}</span>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
            else:
                st.info("ℹ️ No complete paths found between root causes and final effects.")
        except Exception as e:
            st.markdown(
                f"<div class='are-error'>❌ Error computing paths: {e}</div>",
                unsafe_allow_html=True,
            )

    else:
        st.markdown(
            "<div class='are-warning'>"
            "⚠️ No causal relationships could be extracted. "
            "Try a different query or add more relevant data."
            "</div>",
            unsafe_allow_html=True,
        )


# ═══════════════════════════════════════════════════════════════
# TAB 4 — SIMULATE REALITY
# ═══════════════════════════════════════════════════════════════
with tab4:
    if not st.session_state.analysis_done:
        st.markdown(
            "<div class='are-warning'>⚠️ Please complete Steps 2 and 3 first.</div>",
            unsafe_allow_html=True,
        )

    if not st.session_state.api_key:
        st.markdown(
            "<div class='are-warning'>⚠️ Please enter your OpenAI API key in the sidebar.</div>",
            unsafe_allow_html=True,
        )

    if (
        st.session_state.graph is not None
        and st.session_state.graph.number_of_nodes() > 0
    ):
        graph = st.session_state.graph

        st.markdown("## 🔀 Counterfactual Simulation")
        st.markdown("### What if this event never happened?")
        st.markdown(
            "<div class='are-card'>"
            "ℹ️ Select any event from the causal chain below and click <b>Simulate</b> "
            "to see what alternate reality would look like without that event."
            "</div>",
            unsafe_allow_html=True,
        )

        all_nodes = list(graph.nodes())
        chain_nodes = [
            node.strip()
            for node in st.session_state.chain_text.split(" → ")
            if node.strip()
        ]
        selectable_nodes = chain_nodes if chain_nodes else all_nodes
        col_sel, col_ref = st.columns(2)

        with col_sel:
            selected_node = st.selectbox(
                "🎯 Select event to remove (from current displayed chain):",
                options=selectable_nodes,
                key="selectbox_node",
            )

            if selected_node:
                # Show connections
                predecessors = list(graph.predecessors(selected_node))
                successors = list(graph.successors(selected_node))

                st.markdown(
                    f"<div class='are-warning' style='margin-top:0.5rem;'>"
                    f"<b>Removing:</b> {selected_node}"
                    f"</div>",
                    unsafe_allow_html=True,
                )

                preds_str = ", ".join(predecessors) if predecessors else "None (root)"
                succs_str = ", ".join(successors) if successors else "None (final)"
                st.markdown(
                    f"<div class='are-card' style='margin-top:0.5rem;'>"
                    f"<b style='color:#8b949e;'>Causes:</b> "
                    f"<span style='color:#ffd700;'>{preds_str}</span><br>"
                    f"<b style='color:#8b949e;'>Effects:</b> "
                    f"<span style='color:#00bcd4;'>{succs_str}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

            if st.button("🔀 Simulate Alternate Reality", type="primary", key="btn_simulate"):
                if not st.session_state.api_key:
                    st.markdown(
                        "<div class='are-error'>❌ Please enter your OpenAI API key in the sidebar.</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    try:
                        with st.spinner("🌀 Simulating alternate reality..."):
                            result = simulate_counterfactual(
                                graph, selected_node, st.session_state.api_key
                            )
                        st.session_state.counterfactual_result = result
                    except Exception as e:
                        err = str(e).lower()
                        if "invalid api key" in err or "authentication" in err:
                            msg = "Invalid API key. Please check your key."
                        elif "rate limit" in err:
                            msg = "API rate limit reached. Please wait 60 seconds."
                        else:
                            msg = f"Simulation error: {e}"
                        st.markdown(
                            f"<div class='are-error'>❌ {msg}</div>",
                            unsafe_allow_html=True,
                        )

        with col_ref:
            st.markdown("#### 📋 Current Causal Chain")
            if st.session_state.chain_text:
                chain_nodes = st.session_state.chain_text.split(" → ")
                for i, node in enumerate(chain_nodes):
                    arrow = "↓" if i < len(chain_nodes) - 1 else ""
                    badge_class = (
                        "badge-red"
                        if i == 0
                        else ("badge-cyan" if i == len(chain_nodes) - 1 else "badge-purple")
                    )
                    st.markdown(
                        f"<div style='text-align:center;'>"
                        f"<span class='{badge_class}'>{node}</span>"
                        f"</div>"
                        + (
                            f"<div style='text-align:center; color:#8b949e; font-size:1.2rem;'>{arrow}</div>"
                            if arrow
                            else ""
                        ),
                        unsafe_allow_html=True,
                    )
            else:
                st.markdown(
                    "<i style='color:#8b949e;'>Run analysis in Step 2 to see the chain.</i>",
                    unsafe_allow_html=True,
                )

        # ── Results ──
        if st.session_state.counterfactual_result:
            result = st.session_state.counterfactual_result
            st.markdown("---")
            st.markdown("## 🌀 Alternate Reality Results")

            # Summary metrics
            rc1, rc2, rc3 = st.columns(3)
            rc_metrics = [
                ("Removed Event", result.get("removed_node", "—")),
                ("Disconnected Effects", len(result.get("disconnected_effects", []))),
                ("Preserved Causes", len(result.get("ancestor_nodes", []))),
            ]
            for col, (label, val) in zip([rc1, rc2, rc3], rc_metrics):
                with col:
                    st.markdown(
                        f"<div class='metric-card'>"
                        f"<div class='metric-label'>{label}</div>"
                        f"<div class='metric-value' style='font-size:1.1rem;'>{val}</div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

            # Effects / ancestors columns
            eff_col, anc_col = st.columns(2)

            with eff_col:
                st.markdown("#### 📉 What Would NOT Happen:")
                disc = result.get("disconnected_effects", [])
                if disc:
                    badges = "".join(
                        f"<span class='badge-red'>{e}</span>" for e in disc
                    )
                    st.markdown(badges, unsafe_allow_html=True)
                else:
                    st.markdown(
                        "<div class='are-success'>✅ No downstream effects removed.</div>",
                        unsafe_allow_html=True,
                    )

            with anc_col:
                st.markdown("#### 🔍 What Led To This:")
                ancs = result.get("ancestor_nodes", [])
                if ancs:
                    badges = "".join(
                        f"<span class='badge-yellow'>{a}</span>" for a in ancs
                    )
                    st.markdown(badges, unsafe_allow_html=True)
                else:
                    st.markdown(
                        "<i style='color:#8b949e;'>This was a root cause — nothing preceded it.</i>",
                        unsafe_allow_html=True,
                    )

            # LLM explanation
            st.markdown("---")
            st.subheader("🤖 AI Explanation of Alternate Reality")
            explanation = result.get("llm_explanation", "No explanation available.")
            st.markdown(
                f"<div class='llm-box'>{explanation}</div>",
                unsafe_allow_html=True,
            )

            # Modified graph
            st.markdown("---")
            st.subheader(f"🕸️ Modified Causal Graph")
            modified_graph = result.get("modified_graph")
            if modified_graph is not None:
                try:
                    mod_fig = visualize_graph(
                        modified_graph,
                        title=f"Graph After Removing: {result.get('removed_node', '')}",
                    )
                    st.pyplot(mod_fig)
                except Exception as e:
                    st.markdown(
                        f"<div class='are-error'>❌ Error rendering modified graph: {e}</div>",
                        unsafe_allow_html=True,
                    )

            # Download
            st.markdown("---")
            download_text = (
                f"=== Alternate Reality Engine — Counterfactual Report ===\n\n"
                f"Removed Event: {result.get('removed_node', '')}\n\n"
                f"Summary:\n{result.get('summary', '')}\n\n"
                f"Disconnected Effects:\n"
                + "\n".join(f"  - {e}" for e in result.get("disconnected_effects", []))
                + f"\n\nAncestor Causes:\n"
                + "\n".join(f"  - {a}" for a in result.get("ancestor_nodes", []))
                + f"\n\nAI Explanation:\n{result.get('llm_explanation', '')}\n"
            )
            st.download_button(
                label="📥 Download Explanation (.txt)",
                data=download_text,
                file_name="counterfactual_report.txt",
                mime="text/plain",
                key="btn_download",
            )


# ─────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────
st.divider()
fc1, fc2, fc3 = st.columns(3)
with fc1:
    st.markdown(
        "<p class='are-footer'>🌀 Alternate Reality Engine v1.0</p>",
        unsafe_allow_html=True,
    )
with fc2:
    st.markdown(
        "<p class='are-footer'>Built with Streamlit + Groq + In-memory Vector Store</p>",
        unsafe_allow_html=True,
    )
with fc3:
    st.markdown(
        "<p class='are-footer'>Causal RAG for Counterfactual Reasoning</p>",
        unsafe_allow_html=True,
    )
