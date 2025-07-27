import os
import fitz
import pandas as pd
from collections import defaultdict
# REMOVE: from transformers import AutoTokenizer, AutoModel # No longer needemd for MiniLM
from sentence_transformers import SentenceTransformer # NEW import for MiniLM
import torch
import numpy as np # Ensure numpy is imported if not already

# --- 1. Configuration ---
INPUT_DIR = "input"
PDF_FILES = ["file01.pdf", "file02.pdf", "file03.pdf", "file04.pdf", "file05.pdf"]
OUTPUT_FILE = "features.parquet"
MODEL_CHECKPOINT = "my_finetuned_minilm_model" # CHANGED to MiniLM

# Define the embedding dimension for MiniLM-L6-v2
EMBEDDING_DIMENSION = 384

label_map = {
    "Application form for grant of LTC advance": "Title",
    "Overview Foundation Level Extensions": "Title",
    "Revision History": "H1",
    "Table of Contents": "H1",
    "Acknowledgements": "H1",
    "1. Introduction to the Foundation Level Extensions": "H1",
    "2. Introduction to Foundation Level Agile Tester Extension": "H1",
    "2.1 Intended Audience": "H2",
    "2.2 Career Paths for Testers": "H2",
    "2.3 Learning Objectives": "H2",
    "2.4 Entry Requirements": "H2",
    "2.5 Structure and Course Duration": "H2",
    "2.6 Keeping It Current": "H2",
    "3. Overview of the Foundation Level Extension – Agile Tester Syllabus": "H1",
    "3.1 Business Outcomes": "H2",
    "3.2 Content": "H2",
    "4. References": "H1",
    "4.1 Trademarks": "H2",
    "4.2 Documents and Web Sites": "H2",
    "RFP:Request for Proposal To Present a Proposal for Developing the Business Plan for the Ontario Digital Library": "Title",
    "Ontario’s Digital Library": "H1",
    "A Critical Component for Implementing Ontario’s Road Map to Prosperity Strategy": "H1",
    "Summary": "H2",
    "Timeline:": "H3",
    "Background": "H2",
    "Equitable access for all Ontarians:": "H3",
    "Shared decision-making and accountability:": "H3",
    "Shared governance structure:": "H3",
    "Shared funding:": "H3",
    "Local points of entry:": "H3",
    "Access:": "H3",
    "Guidance and Advice:": "H3",
    "Training:": "H3",
    "Provincial Purchasing & Licensing:": "H3",
    "Technological Support:": "H3",
    "What could the ODL really mean?": "H3",
    "The Business Plan to be Developed": "H2",
    "Milestones": "H3",
    "Approach and Specific Proposal Requirements": "H2",
    "Evaluation and Awarding of Contract": "H2",
    "Appendix A: ODL Envisioned Phases & Funding": "H2",
    "Phase I: Business Planning": "H3",
    "Phase II: Implementing and Transitioning": "H3",
    "Phase III: Operating and Growing the ODL": "H3",
    "Appendix B: ODL Steering Committee Terms of Reference": "H2",
    "1. Preamble": "H3",
    "2. Terms of Reference": "H3",
    "3. Membership": "H3",
    "4. Appointment Criteria and Process": "H3",
    "5. Term": "H3",
    "6. Chair": "H3",
    "7. Meetings": "H3",
    "8. Lines of Accountability and Communication": "H3",
    "9. Financial and Administrative Policies": "H3",
    "Appendix C: ODL’s Envisioned Electronic Resources": "H2",
    "Parsippany -Troy Hills STEM Pathways": "Title",
    "PATHWAY OPTIONS": "H1",
    "HOPE To SEE You THERE!": "H1",
}

# --- CHANGED: Load SentenceTransformer model ---
# tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT) # REMOVE
# model = AutoModel.from_pretrained(MODEL_CHECKPOINT) # REMOVE
embedding_model = SentenceTransformer(MODEL_CHECKPOINT) # NEW: Load SentenceTransformer model
def get_text_embedding(text):
    """Generates sentence embeddings for a given text using SentenceTransformer."""
    if not text.strip():
        # Return a zero vector for empty or whitespace-only text
        return np.zeros(EMBEDDING_DIMENSION, dtype=np.float32) # USE EMBEDDING_DIMENSION

    # The SentenceTransformer model's encode method handles tokenization and pooling
    embedding = embedding_model.encode(text, convert_to_numpy=True, show_progress_bar=False)
    return embedding
    
all_lines_data = []
for pdf_file in PDF_FILES:
    pdf_path = os.path.join(INPUT_DIR, pdf_file)
    print(f"Processing and engineering features for {pdf_path}...")
    doc = fitz.open(pdf_path)
    
    page_font_stats = {}
    for page in doc:
        font_sizes = [span['size'] for block in page.get_text("dict")["blocks"] if block.get('type')==0 for line in block['lines'] for span in line['spans']]
        if font_sizes:
            page_font_stats[page.number] = {'avg': sum(font_sizes) / len(font_sizes), 'max': max(font_sizes)}
        else:
            page_font_stats[page.number] = {'avg': 10, 'max': 12}

    for page_num, page in enumerate(doc):
        lines = defaultdict(list)
        blocks = page.get_text("dict")["blocks"]
        for b in blocks:
            if b.get('type') == 0:
                for l in b['lines']:
                    y0 = round(l['bbox'][1])
                    for s in l['spans']:
                        lines[y0].append(s)
        
        sorted_lines = sorted(lines.items())
        prev_y1 = 0
        for _, spans in sorted_lines:
            if not spans: continue
            spans.sort(key=lambda s: s['bbox'][0])
            full_text = " ".join([s['text'].strip() for s in spans]).strip()
            if not full_text: continue

            first_span = spans[0]
            y0 = min(s['bbox'][1] for s in spans)
            y1 = max(s['bbox'][3] for s in spans)

            all_lines_data.append({
                "text": full_text, "label": label_map.get(full_text, "Body"),
                "font_size": first_span["size"], "is_bold": "bold" in first_span["font"].lower(),
                "relative_font_size": first_span["size"] / page_font_stats[page_num]['avg'],
                "pos_y": y0 / page.rect.height, "space_above": y0 - prev_y1,
                "word_count": len(full_text.split()),
            })
            prev_y1 = y1
    doc.close()

df = pd.DataFrame(all_lines_data)
print("Generating text embeddings...")
df['text_embedding'] = df['text'].apply(get_text_embedding)
df.to_parquet(OUTPUT_FILE)
print(f"\nSuccessfully created feature-rich dataset at '{OUTPUT_FILE}'")