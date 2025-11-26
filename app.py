import streamlit as st
import asyncio
import os
import pandas as pd
import networkx as nx
import gc
import sys
import warnings
from datetime import datetime
from hyde_query_3 import LegalSearchEngine 

# --- CONFIG & SUPPRESSION ---
st.set_page_config(page_title="BNS Legal Expert", layout="wide")
GRAPH_PATH = "./bns_legal_graph/graph_chunk_entity_relation.graphml"

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["GRPC_VERBOSITY"] = "ERROR"

# --- LOGGER ---
def log(msg, icon="🖥️"):
    timestamp = datetime.now().strftime("%H:%M:%S")
    # Using 'flush=True' ensures logs appear immediately in the terminal
    print(f"[{timestamp}] {icon} {msg}", flush=True)

# --- HELPER: Async Runner ---
def run_async(coro):
    """
    Safely run an async coroutine from Streamlit (which is sync code).
    Avoids using asyncio.run() repeatedly and handles missing event loop.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)

# --- CACHE DATA ---
@st.cache_resource
def load_raw_data():
    log("Streamlit Cache Miss: Loading Graph Data...", "💾")
    
    if not os.path.exists(GRAPH_PATH):
        log(f"Graph Not Found at {GRAPH_PATH}", "❌")
        return None, None, None

    graph = nx.read_graphml(GRAPH_PATH)
    temp_engine = LegalSearchEngine(GRAPH_PATH)
    
    log("Building Embeddings for Cache...", "🧱")
    with st.spinner("🧠 Building Neural Index (One-time setup)..."):
        run_async(temp_engine.build_runtime_index())
    
    gc.collect()
    log("Graph Data Cached Successfully.", "💾")
    return graph, temp_engine.node_embeddings, temp_engine.node_names

# --- INITIALIZATION ---
log("Initializing UI Session...", "🚀")
st.title("⚖️ BNS Legal Expert AI")
st.caption("Full BNS Knowledge Graph • Memory Enabled • Critic Verified")

graph_data, embeddings, node_names = load_raw_data()

if graph_data is not None:
    current_engine = LegalSearchEngine(GRAPH_PATH)
    current_engine.graph = graph_data
    current_engine.node_embeddings = embeddings
    current_engine.node_names = node_names
    st.sidebar.success(f"Graph Active: {len(node_names)} Nodes")
else:
    st.error(f"Graph file not found at {GRAPH_PATH}")
    current_engine = None

# --- GRAPH INSPECTOR (Sidebar) ---
with st.sidebar:
    st.markdown("---")
    st.header("Graph Inspector 🕵️")
    node_search = st.text_input("Check if node exists:", placeholder="e.g. Section 303")
    
    if node_search and current_engine:
        log(f"User inspecting node: {node_search}", "🔍")
        found = False
        for node in current_engine.node_names:
            if node_search.lower() in node.lower():
                st.write(f"✅ Found: **{node}**")
                found = True
                log(f"Inspector found: {node}", "✅")
        if not found:
            st.error(f"❌ Node '{node_search}' NOT found.")
            log(f"Inspector failed to find: {node_search}", "❌")

# --- CHAT HISTORY STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- RENDER CHAT HISTORY ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # Critic display
        if "critique" in message:
            crit = message["critique"]
            if isinstance(crit, str):
                if crit.startswith("PASS (Reasoned Extension"):
                    st.success(f"✅ Verified (with extra reasoning): {crit}")
                elif crit.startswith("PASS"):
                    st.success(f"✅ Verified: {crit}")
                elif "Honest Gap" in crit:
                    st.info(f"ℹ️ Honest Gap: {crit}")
                elif "FAIL" in crit:
                    st.warning(f"⚠️ Verification Failed: {crit}")
                else:
                    st.info(f"ℹ️ {crit}")

        # Debug table (legal evidence)
        if "debug_data" in message:
            with st.expander("View Legal Evidence"):
                st.dataframe(message["debug_data"], hide_index=True)

# --- MAIN CHAT INPUT / EXECUTION ---
if prompt := st.chat_input("Describe a situation or ask about a BNS section..."):
    log(f"User Prompt Received: {prompt}", "👤")
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if current_engine:
            message_placeholder = st.empty()
            message_placeholder.markdown("🔍 Consulting Knowledge Graph & LLM...")

            try:
                # --- MEMORY STRING (last few turns) ---
                history_str = ""
                for msg in st.session_state.messages[-5:]:
                    role_name = "User" if msg["role"] == "user" else "Bot"
                    history_str += f"{role_name}: {msg['content']}\n"
                
                log("Sending Query to Engine...", "📡")
                
                # --- EXECUTE SEARCH PIPELINE ---
                answer, context_str, critique = run_async(
                    current_engine.search(prompt, chat_history=history_str)
                )
                
                log("Response Received from Engine.", "📥")
                message_placeholder.markdown(answer)
                
                # --- CRITIC DISPLAY ---
                if isinstance(critique, str):
                    if critique.startswith("PASS (Reasoned Extension"):
                        st.success(f"✅ Verified (with extra reasoning): {critique}")
                    elif critique.startswith("PASS"):
                        st.success(f"✅ Verified: {critique}")
                    elif "Honest Gap" in critique:
                        st.info(f"ℹ️ Honest Gap: {critique}")
                    elif "FAIL" in critique:
                        st.warning(f"⚠️ Verification Failed: {critique}")
                    else:
                        st.info(f"ℹ️ {critique}")

                # --- DEBUG TABLE (build from context_str) ---
                debug_rows = []
                current_source = ""
                for line in context_str.split('\n'):
                    if line.startswith("SOURCE:"):
                        current_source = line.replace("SOURCE: ", "").strip()
                    elif line.startswith(" -> LINKED SECTIONS:"):
                        sections = line.replace(" -> LINKED SECTIONS: ", "").split(", ")
                        for s in sections:
                            debug_rows.append({
                                "Concept": current_source,
                                "Type": "Linked Law",
                                "Detail": s
                            })
                    elif "PENALTY:" in line:
                        penalty = line.split("PENALTY: ")[1]
                        debug_rows.append({
                            "Concept": current_source,
                            "Type": "Punishment",
                            "Detail": penalty
                        })
                
                if debug_rows:
                    df = pd.DataFrame(debug_rows)
                    with st.expander("View Legal Evidence"):
                        st.dataframe(df, hide_index=True)
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "debug_data": df,
                        "critique": critique
                    })
                else:
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "critique": critique
                    })
                
                log("UI Update Complete.", "🏁")
            
            except Exception as e:
                log(f"Runtime Error: {e}", "🔥")
                message_placeholder.error(f"Error: {e}")
        else:
            st.error("Engine not ready.")
