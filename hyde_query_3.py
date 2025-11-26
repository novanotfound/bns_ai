import os
import asyncio
import numpy as np
import networkx as nx
import google.generativeai as genai
import time
from datetime import datetime
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

# --- CONFIGURATION ---
GRAPH_PATH = "./bns_legal_graph/graph_chunk_entity_relation.graphml"

# --- ENV SETUP ---
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("❌ GOOGLE_API_KEY not found")
genai.configure(api_key=api_key)

# --- LOGGER UTILS ---
def log(msg, icon="ℹ️"):
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {icon} {msg}")


class LegalSearchEngine:
    def __init__(self, graph_path):
        log(f"Loading Knowledge Graph from {graph_path}...", "📂")
        self.graph = nx.read_graphml(graph_path)
        log(f"Graph Loaded: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges.", "✅")
        self.node_embeddings = None
        self.node_names = []
        
    def get_lock(self):
        return asyncio.Semaphore(1)

    async def _safe_embed(self, texts, lock=None):
        if lock is None:
            lock = self.get_lock()
        start_t = time.time()
        
        async with lock:
            try:
                log(f"Embedding {len(texts)} text chunks...", "🧠")
                await asyncio.sleep(1.5)  # Safety buffer
                result = await asyncio.to_thread(
                    genai.embed_content,
                    model="models/text-embedding-004",
                    content=texts,
                    task_type="retrieval_query"
                )
                
                duration = time.time() - start_t
                log(f"Embedding complete ({duration:.2f}s)", "⚡")
                
                if "embedding" in result:
                    # Single or batched response handling
                    if isinstance(result["embedding"][0], list):
                        return np.array(result["embedding"])
                    return np.array([result["embedding"]])
                return np.zeros((len(texts), 768))
            except Exception as e:
                log(f"Embedding Error: {e}", "❌")
                return np.zeros((len(texts), 768))

    async def _safe_generate(self, prompt, task_name="Generation", lock=None):
        """
        Safe LLM call wrapper.

        IMPORTANT: Uses synchronous generate_content() inside asyncio.to_thread()
        to avoid grpc.aio event-loop issues when running under Streamlit.
        """
        if lock is None:
            lock = self.get_lock()

        model = genai.GenerativeModel("models/gemini-2.5-flash")
        start_t = time.time()

        log(f"Starting LLM Task: {task_name}...", "🤖")

        async with lock:
            try:
                # No fixed sleep here; if you want rate limiting, handle 429s explicitly.
                # Run blocking LLM call in a thread to keep this coroutine non-blocking.
                response = await asyncio.to_thread(
                    model.generate_content,
                    prompt
                )

                duration = time.time() - start_t
                text = response.text or ""
                preview = text[:50].replace("\n", " ") + "..."
                log(f"LLM Success ({duration:.2f}s): {preview}", "💬")

                return text

            except Exception as e:
                log(f"LLM Error in {task_name}: {e}", "❌")
                return f"Error: {e}"

    async def build_runtime_index(self):
        log("Building Context-Aware Vector Index...", "⚙️")
        build_lock = self.get_lock()
        
        nodes = list(self.graph.nodes(data=True))
        self.node_names = [n[0] for n in nodes]
        rich_texts = []
        
        # Enrich Context
        for node_name, data in nodes:
            text = f"ENTITY: {node_name}. TYPE: {data.get('type', 'Unknown')}. DESC: {data.get('description', '')}. "
            neighbors = list(self.graph.neighbors(node_name))
            if neighbors:
                neighbor_str = ", ".join(neighbors[:5])
                text += f" RELATES TO: {neighbor_str}"
            rich_texts.append(text)
        
        log(f"Prepared {len(rich_texts)} nodes for indexing.", "📝")
        
        # Batch Embed
        BATCH_SIZE = 20
        embeddings = []
        total = len(rich_texts)
        
        for i in range(0, total, BATCH_SIZE):
            batch = rich_texts[i : i + BATCH_SIZE]
            if i % 100 == 0:
                log(f"Processing batch {i}/{total}...", "🔄")
            embs = await self._safe_embed(batch, lock=build_lock)
            embeddings.append(embs)
            
        self.node_embeddings = np.vstack(embeddings)
        log("Vector Index Ready in Memory.", "✅")

    # --- CRITIC MODULE ---
    async def critique_answer(self, user_query, answer, context_str, lock):
        log("Summoning the Critic...", "👨‍⚖️")
        critic_prompt = f"""
        You are a Legal Auditor. Verify the Generated Answer against the Retrieved BNS Context.

        1. RETRIEVED CONTEXT:
        {context_str}

        2. GENERATED ANSWER:
        {answer}

        TASK:
        - Focus on legal facts: sections, offences, penalties, definitions.
        - Ignore purely illustrative examples as long as they are clearly marked as examples and do not contradict the law.

        You must choose exactly ONE label:
        - "PASS"                 → All legal facts are supported by or consistent with the context.
        - "PASS (Reasoned Extension)" → The answer adds extra explanation or examples beyond the context, 
                                       but does not contradict any legal fact in the context.
        - "FAIL: Hallucinated Citation" → The answer invents legal sections/penalties/rules that do NOT appear in the context.
        - "FAIL: Contradiction" → The answer clearly conflicts with the context for some legal fact.

        OUTPUT:
        Return only the label, nothing else.
        """
        return await self._safe_generate(critic_prompt, task_name="Critic Review", lock=lock)

    async def search(self, user_query, chat_history=""):
        search_lock = self.get_lock()
        log(f"New Query Received: {user_query}", "📥")
        
        # 1. HyDE: build a hypothetical section to improve retrieval
        full_prompt = f"""
        CHAT HISTORY:
        {chat_history}

        CURRENT QUERY: "{user_query}"

        TASK:
        Write a short, hypothetical BNS-style legal provision that would answer this query.
        Do NOT include analysis or explanation here; just a concise legal-style section.
        """
        fake_law = await self._safe_generate(full_prompt, task_name="HyDE Generation", lock=search_lock)
        
        # 2. Embed Query
        query_emb = await self._safe_embed([fake_law], lock=search_lock)
        
        # 3. Search in vector space
        log("Searching Vector Space...", "🔍")
        similarities = cosine_similarity(query_emb, self.node_embeddings)[0]
        top_indices = np.argsort(similarities)[-8:][::-1]
        top_nodes = [self.node_names[i] for i in top_indices]
        
        log(f"Top 8 Nodes Found: {top_nodes}", "🎯")
        
        # 4. Build context from graph
        context = []
        for node_name in top_nodes:
            node_data = self.graph.nodes[node_name]
            node_info = f"SOURCE: {node_name} ({node_data.get('type', 'Unknown')})\nSUMMARY: {node_data.get('description', '')}"
            neighbors = self.graph.neighbors(node_name)
            connected_sections = []
            for nb in neighbors:
                if self.graph.nodes[nb].get("type") == "SECTION":
                    connected_sections.append(nb)
                if self.graph.nodes[nb].get("type") == "PENALTY":
                    node_info += f"\n -> PENALTY: {nb}"
            if connected_sections:
                node_info += f"\n -> LINKED SECTIONS: {', '.join(connected_sections)}"
            context.append(node_info)
        
        context_str = "\n---\n".join(context)

        # 5. STAGE 1: Draft answer (free reasoning, no hard context constraint)
        draft_prompt = f"""
        You are a legal assistant specializing in Indian criminal law.

        USER QUERY:
        {user_query}

        TASK:
        1. Answer the user's query in clear, simple language.
        2. Provide 1–2 concrete, intuitive examples. 
           Label each example explicitly as: "Illustrative Example".
        3. You may rely on your general legal knowledge.
        4. If you mention any specific section numbers or punishments, do so only if you are reasonably confident.
        """
        draft_answer = await self._safe_generate(draft_prompt, task_name="Draft Answer", lock=search_lock)

        # 6. STAGE 2: Refine using graph context as ground truth
        refine_prompt = f"""
        You are now a BNS legal expert.

        You are given:

        A) DRAFT ANSWER (may contain mistakes):
        {draft_answer}

        B) RETRIEVED BNS CONTEXT (canonical legal ground truth):
        {context_str}

        C) USER QUERY:
        {user_query}

        TASK:
        1. Correct any legal mistakes in the DRAFT ANSWER using ONLY the BNS CONTEXT as the ground truth 
           for sections, offences, and penalties.
        2. You MAY keep intuitive explanations and "Illustrative Example" content, 
           as long as they do not contradict the context.
        3. Do NOT invent any new section numbers, offences, or punishments that are not supported by the context.
        4. If the DRAFT ANSWER mentions a specific section or penalty that is NOT present in the context:
            - Either remove it, OR
            - Rephrase it as a general explanation without claiming it is an exact BNS rule.
        5. At the end of your answer, add a short section titled "BNS References" 
           where you list relevant SOURCE nodes and any sections/penalties clearly supported by the context.

        OUTPUT:
        Provide the final, corrected answer ready for the user.
        """
        answer = await self._safe_generate(refine_prompt, task_name="Final Answer", lock=search_lock)
        
        # 7. Critique with updated labels
        critique = await self.critique_answer(user_query, answer, context_str, search_lock)
        
        return answer, context_str, critique
