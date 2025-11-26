import pandas as pd
import json
import os

# --- CONFIGURATION ---
INPUT_CSV_PATH = "bns_sections.csv"
OUTPUT_JSON_PATH = "bns_smart_chunks.json"

# Column mapping (Update these strings to match your exact CSV headers)
COL_CHAPTER = "Chapter" 
COL_CHAPTER_NAME = "Chapter_name"
# [NEW] Added Subtype
COL_CHAPTER_SUBTYPE = "Chapter_subtype" 
COL_SECTION = "Section" 
COL_SECTION_NAME = "Section _name" 
COL_DESCRIPTION = "Description" 

def build_smart_chunk(row):
    """
    Constructs a context-rich string for a single legal section.
    Includes Chapter Subtype to give granular context to the LLM.
    """
    # clean strings and handle missing values
    chapter_num = str(row.get(COL_CHAPTER, "Unknown")).strip()
    chapter_title = str(row.get(COL_CHAPTER_NAME, "")).strip()
    # [NEW] Extract subtype (e.g. "Of Sexual Offences")
    chapter_subtype = str(row.get(COL_CHAPTER_SUBTYPE, "")).strip()
    section_num = str(row.get(COL_SECTION, "Unknown")).strip()
    section_title = str(row.get(COL_SECTION_NAME, "")).strip()
    legal_text = str(row.get(COL_DESCRIPTION, "")).strip()

    # If subtype exists, we add it to the text block. 
    # If it's "nan" or empty, we just leave it out to save tokens.
    subtype_line = ""
    if chapter_subtype and chapter_subtype.lower() != "nan":
        subtype_line = f"CHAPTER_SUBTYPE: {chapter_subtype}\n"

    # The Template
    # We inject the subtype right after the chapter title
    chunk_text = (
        f"CHAPTER_ID: {chapter_num}\n"
        f"CHAPTER_TITLE: {chapter_title}\n"
        f"{subtype_line}" 
        f"SECTION_ID: {section_num}\n"
        f"SECTION_TITLE: {section_title}\n"
        f"LEGAL_PROVISION:\n{legal_text}\n"
    )
    
    return {
        "id": section_num,
        "title": section_title,
        "content": chunk_text,
        "metadata": {
            "chapter": chapter_num,
            "chapter_subtype": chapter_subtype, # Added to metadata
            "section": section_num
        }
    }

def main():
    print(f"🔍 Loading data from {INPUT_CSV_PATH}...")
    
    if not os.path.exists(INPUT_CSV_PATH):
        print(f"❌ Error: File not found at {INPUT_CSV_PATH}")
        return

    try:
        df = pd.read_csv(INPUT_CSV_PATH)
        print(f"✅ Loaded {len(df)} rows.")
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        return

    # --- VALIDATION STEP ---
    initial_count = len(df)
    
    # Drop rows where the critical 'Description' is missing
    df = df.dropna(subset=[COL_DESCRIPTION])
    # Filter out very short/invalid descriptions
    df = df[df[COL_DESCRIPTION].astype(str).str.len() > 10]
    
    dropped_count = initial_count - len(df)
    if dropped_count > 0:
        print(f"⚠️ Dropped {dropped_count} invalid rows.")
    
    print(f"⚙️ Processing {len(df)} valid legal sections...")

    # --- TRANSFORMATION STEP ---
    smart_chunks = []
    
    for _, row in df.iterrows():
        chunk_obj = build_smart_chunk(row)
        smart_chunks.append(chunk_obj)

    # --- EXPORT STEP ---
    try:
        with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(smart_chunks, f, indent=4, ensure_ascii=False)
        
        print(f"✅ Success! Data exported to {OUTPUT_JSON_PATH}")
        print("SAMPLE CHUNK PREVIEW:\n" + "-"*40)
        print(smart_chunks[0]['content'])
        print("-" * 40)
        
    except Exception as e:
        print(f"❌ Error saving JSON: {e}")

if __name__ == "__main__":
    main()