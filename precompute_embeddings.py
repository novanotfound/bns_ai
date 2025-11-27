import json
import os
import numpy as np
import networkx as nx

from dotenv import load_dotenv
import google.generativeai as genai

GRAPH_PATH = "./bns_legal_graph/graph_chunk_entity_relation.graphml"

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found")
genai.configure(api_key=api_key)

def build_rich_texts(graph):
    texts = []
    names = []
    for node_name, data in graph.nodes(data=True):
        text = f"ENTITY: {node_name}. TYPE: {data.get('type', 'Unknown')}. DESC: {data.get('description', '')}. "
        neighbors = list(graph.neighbors(node_name))
        if neighbors:
            neighbor_str = ", ".join(neighbors[:5])
            text += f" RELATES TO: {neighbor_str}"
        texts.append(text)
        names.append(node_name)
    return names, texts

def main():
    if not os.path.exists(GRAPH_PATH):
        raise FileNotFoundError(f"Graph not found at {GRAPH_PATH}")

    print("📂 Loading graph...")
    graph = nx.read_graphml(GRAPH_PATH)

    print("🧱 Building rich texts for nodes...")
    node_names, rich_texts = build_rich_texts(graph)

    print(f"🧠 Embedding {len(rich_texts)} nodes...")
    # simple sync embedding in batches
    BATCH_SIZE = 20
    all_embs = []
    for i in range(0, len(rich_texts), BATCH_SIZE):
        batch = rich_texts[i : i + BATCH_SIZE]
        print(f"  -> batch {i}/{len(rich_texts)}")
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=batch,
            task_type="retrieval_document",
        )
        # handle batch vs single
        if "embedding" in result:
            embs = result["embedding"]
        else:
            # older clients
            embs = [r["embedding"] for r in result]
        all_embs.extend(embs)

    embeddings = np.array(all_embs)
    print("💾 Saving node_embeddings.npy and node_names.json ...")
    np.save("node_embeddings.npy", embeddings)
    with open("node_names.json", "w", encoding="utf-8") as f:
        json.dump(node_names, f, ensure_ascii=False, indent=2)

    print("🎉 Done. Precomputed embeddings are ready.")

if __name__ == "__main__":
    main()
