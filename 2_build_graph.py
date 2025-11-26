import json
import os
import shutil
import asyncio
import numpy as np
import google.generativeai as genai
from dotenv import load_dotenv
import networkx as nx
import hashlib

# --- CONFIGURATION ---
WORKING_DIR = "./bns_legal_graph"
INPUT_JSON_PATH = "bns_smart_chunks.json"
CHECKPOINT_FILE = os.path.join(WORKING_DIR, "extraction_checkpoint.json")

# --- LOAD API KEY ---
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("❌ GOOGLE_API_KEY not found")

genai.configure(api_key=api_key)

# --- RATE LIMITING ---
RATE_LIMIT_LOCK = asyncio.Semaphore(1)  # Strict safety
api_call_count = 0

# --- HELPER: CALL GEMINI ---
async def call_gemini(prompt: str) -> str:
    global api_call_count
    model = genai.GenerativeModel('models/gemini-2.5-flash')
    
    async with RATE_LIMIT_LOCK:
        for attempt in range(5):
            try:
                # 4-second delay for Free Tier safety
                await asyncio.sleep(2)
                api_call_count += 1
                response = await model.generate_content_async(prompt)
                return response.text.strip()
            except Exception as e:
                if "429" in str(e):
                    print(f"⏳ Rate limit. Sleeping {30 * (attempt+1)}s...")
                    await asyncio.sleep(30 * (attempt + 1))
                else:
                    return ""
    return ""

# --- 3-PASS EXTRACTION ---
async def extract_entities(text: str) -> list:
    """Pass 1: Extract entities"""
    prompt = f"""Extract legal entities from this text.
TYPES: SECTION, OFFENSE, PENALTY, CHAPTER
OUTPUT: One per line in format: Name | Type
TEXT: {text[:2000]}
ENTITIES:"""
    response = await call_gemini(prompt)
    entities = []
    for line in response.split('\n'):
        if '|' in line:
            parts = line.split('|', 1)
            if len(parts) == 2:
                name = parts[0].strip()
                ent_type = parts[1].strip().upper()
                if ent_type in ['SECTION', 'OFFENSE', 'PENALTY', 'CHAPTER', 'EXCEPTION']:
                    entities.append({
                        'name': name,
                        'type': ent_type,
                        'id': hashlib.md5(name.encode()).hexdigest()[:16]
                    })
    return entities

async def extract_relationships(text: str, entities: list) -> list:
    """Pass 2: Extract relationships"""
    if len(entities) < 2:
        return []
    entity_list = "\n".join([f"- {e['name']}" for e in entities])
    prompt = f"""Find connections between these entities.
ENTITIES:
{entity_list}
TYPES: defines, punishes, contains, relates_to
OUTPUT: One per line in format: Source | Type | Target
CONTEXT: {text[:1500]}
RELATIONSHIPS:"""
    response = await call_gemini(prompt)
    relationships = []
    entity_names = {e['name'] for e in entities}
    for line in response.split('\n'):
        if '|' in line:
            parts = line.split('|')
            if len(parts) == 3:
                source = parts[0].strip()
                rel_type = parts[1].strip().lower()
                target = parts[2].strip()
                if source in entity_names and target in entity_names:
                    relationships.append({'source': source, 'target': target, 'type': rel_type})
    return relationships

async def process_chunk(text: str, section_id: str, section_title: str = "") -> dict:
    """
    Process one legal chunk:
    1. Extract entities via LLM
    2. Force-add a SECTION node like "Section 303" (or "Section 303: Theft ...")
    3. Extract relationships including this section.
    """
    entities = await extract_entities(text)
    if not entities:
        entities = []

    # Build a nice readable node name
    if section_title:
        section_entity_name = f"Section {section_id}: {section_title}"
    else:
        section_entity_name = f"Section {section_id}"

    # Avoid duplicate if LLM already created this entity name
    if not any(e["name"].lower() == section_entity_name.lower() for e in entities):
        entities.append({
            "name": section_entity_name,
            "type": "SECTION",
            "id": hashlib.md5(section_entity_name.encode()).hexdigest()[:16]
        })

    relationships = await extract_relationships(text, entities) if entities else []
    return {'section_id': section_id, 'entities': entities, 'relationships': relationships}

# --- SAVE FUNCTIONS ---
def save_graph_to_disk(results, directory):
    """Saves the graph in GraphML format for Step 3"""
    G = nx.Graph()
    
    # 1. Add Nodes
    for res in results:
        for ent in res['entities']:
            G.add_node(ent['name'], type=ent['type'], description=ent.get('description', ''))

    # 2. Add Edges
    for res in results:
        for rel in res['relationships']:
            G.add_edge(rel['source'], rel['target'], label=rel['type'])
    
    # 3. Save as GraphML (Standard format)
    output_path = os.path.join(directory, "graph_chunk_entity_relation.graphml")
    nx.write_graphml(G, output_path)
    print(f"💾 Graph saved to {output_path} ({G.number_of_nodes()} nodes)")

async def main():
    # Setup Directory
    if os.path.exists(WORKING_DIR):
        shutil.rmtree(WORKING_DIR)
    os.makedirs(WORKING_DIR, exist_ok=True)
    
    # Load Data
    if not os.path.exists(INPUT_JSON_PATH):
        print(f"❌ {INPUT_JSON_PATH} not found")
        return
    with open(INPUT_JSON_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"🚀 Starting Extraction for {len(chunks)} sections...")
    print("🔒 Saving progress every 10 chunks.")

    results = []
    
    for i, chunk in enumerate(chunks, 1):
        section_id = chunk.get('id', '')
        section_title = chunk.get('title', '')
        text = chunk.get('content', '')

        print(f"[{i}/{len(chunks)}] Processing Section {section_id}...", end=" ", flush=True)
        
        try:
            result = await process_chunk(text, section_id, section_title)
            results.append(result)
            
            ent_cnt = len(result['entities'])
            rel_cnt = len(result['relationships'])
            print(f"✅ ({ent_cnt} ents, {rel_cnt} rels)")
            
            # --- INCREMENTAL SAVE ---
            if i % 10 == 0 or i == len(chunks):
                # Save Raw JSON
                with open(CHECKPOINT_FILE, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                # Save GraphML (So Step 3 works even if we stop early)
                save_graph_to_disk(results, WORKING_DIR)
                
        except Exception as e:
            print(f"❌ Error: {e}")

    print("\n🎉 ALL DONE!")

if __name__ == "__main__":
    asyncio.run(main())
