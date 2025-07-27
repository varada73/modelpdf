import os
import json
import fitz
import pandas as pd
import numpy as np
import joblib
import torch
from collections import defaultdict
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import datetime
import re
import nltk
from nltk.tokenize import sent_tokenize

# Download NLTK punkt tokenizer data if not present (do this during Docker build for offline use)
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt', quiet=True) # Ensure this is done in Dockerfile or pre-downloaded

# --- 1. Configuration & Model Loading (from Round 1A) ---
MODEL_DIR = "." # Assuming models are in the same directory as the script
MODEL_FILE = os.path.join(MODEL_DIR, "lgbm_classifier.joblib")
FEATURES_FILE = os.path.join(MODEL_DIR, "lgbm_features.json")
LABEL_MAP_FILE = os.path.join(MODEL_DIR, "label_map.json")
INPUT_DIR = "sample_input2" # Changed from "input" to be more specific
OUTPUT_DIR = "output_results2" # Changed from "output" to be more specific
BERT_CHECKPOINT = "my_finetuned_minilm_model" # Keep the existing BERT model

# --- NEW: Similarity Threshold for extracted sections/subsections ---
MIN_SIMILARITY_THRESHOLD = 0.10 # Set your desired minimum similarity (0.0 to 1.0)

# Placeholder for Round 1B specific inputs (will be read from files in a real setup)
# For demonstration purposes, hardcoding these.
PERSONA_DEFINITION = "HR professional"
JOB_TO_BE_DONE = "Create and manage fillable forms for onboarding and compliance."

# Load pre-trained models and data
lgbm_classifier = joblib.load(MODEL_FILE)
with open(FEATURES_FILE, 'r') as f:
    model_features = json.load(f)
with open(LABEL_MAP_FILE, 'r') as f:
    id2label = json.load(f)['id2label']
    id2label = {int(k): v for k, v in id2label.items()}

model = SentenceTransformer(BERT_CHECKPOINT)
EMBEDDING_DIMENSION=384
# Function to get text embeddings (from Round 1A)

def get_text_embedding(text):
    """Generates sentence embeddings for a given text using SentenceTransformer."""
    if not text.strip():
        # Return a zero vector for empty or whitespace-only text
        # Ensure EMBEDDING_DIMENSION is defined globally, e.g., 384 for MiniLM-L6-v2
        return np.zeros(EMBEDDING_DIMENSION, dtype=np.float32)

    # The SentenceTransformer model's encode method handles tokenization and pooling
    # convert_to_numpy=True ensures the output is a numpy array
    embedding = model.encode(text, convert_to_numpy=True, show_progress_bar=False)
    return embedding

# --- 2. Core PDF Processing Functions (Modified from Round 1A) ---

def extract_blocks_and_features(pdf_path):
    """
    Extracts text blocks and computes visual/structural features from a PDF.
    Returns a pandas DataFrame where each row is a detected text block.
    """
    doc = fitz.open(pdf_path)
    all_blocks_data = []
    page_font_stats = {}

    # Calculate average and max font sizes per page for relative features
    for page in doc:
        font_sizes = [span['size'] for block in page.get_text("dict")["blocks"] if block.get('type')==0 for line in block['lines'] for span in line['spans']]
        if font_sizes:
            page_font_stats[page.number] = {'avg': sum(font_sizes) / len(font_sizes), 'max': max(font_sizes)}
        else:
            page_font_stats[page.number] = {'avg': 10, 'max': 12} # Default if page has no text content

    for page_num, page in enumerate(doc):
        blocks_on_page = page.get_text("dict")["blocks"]

        prev_block_y1 = 0 # Track the end Y-coordinate of the previous block for 'space_above'

        for b_idx, b in enumerate(blocks_on_page):
            if b.get('type') == 0: # Process only text blocks (type 0)
                full_block_text = ""
                block_min_y0 = float('inf')
                block_max_y1 = 0
                font_size_sum = 0
                font_count = 0
                is_bold_flag = False

                # Iterate through lines and spans within the current block to get text and features
                for l in b['lines']:
                    line_text = ""
                    for s in l['spans']:
                        line_text += s['text'] # Accumulate text for the line
                        font_size_sum += s['size']
                        font_count += 1
                        if "bold" in s["font"].lower():
                            is_bold_flag = True
                    full_block_text += line_text.strip() + " " # Add line text to block text
                    block_min_y0 = min(block_min_y0, l['bbox'][1])
                    block_max_y1 = max(block_max_y1, l['bbox'][3])

                full_block_text = full_block_text.strip() # Final strip for the block

                if not full_block_text:
                    continue # Skip empty blocks

                avg_font_size = font_size_sum / font_count if font_count > 0 else 0

                all_blocks_data.append({
                    "text": full_block_text,
                    "font_size": avg_font_size, # Use average font size for the block
                    "is_bold": is_bold_flag,
                    "relative_font_size": avg_font_size / page_font_stats[page_num]['avg'],
                    "pos_y": block_min_y0 / page.rect.height, # Relative Y position
                    "space_above": block_min_y0 - prev_block_y1, # Space from previous block
                    "word_count": len(full_block_text.split()),
                    "page_num": page_num + 1, # 1-indexed page number
                    "y0_abs": block_min_y0, # Absolute Y-start position
                    "y1_abs": block_max_y1 # Absolute Y-end position
                })
                prev_block_y1 = block_max_y1 # Update for the next block's space_above

    doc.close()
    return pd.DataFrame(all_blocks_data)


def classify_blocks_and_build_outline(pdf_df_blocks, doc_title_fallback=""):
    """
    Takes a DataFrame of blocks with features, classifies them using the LGBM model,
    and constructs the hierarchical document outline (Round 1A output).
    Returns the document title, hierarchical outline, and the DataFrame with classified blocks.
    """
    if pdf_df_blocks.empty:
        return doc_title_fallback, [], pd.DataFrame()

    # Generate text embeddings for all blocks
    text_embeddings = np.array(pdf_df_blocks['text'].apply(get_text_embedding).tolist())

    # Prepare visual features for classification
    # Dynamically create BERT embedding feature names to exclude from visual_features selection
    bert_embedding_dim = EMBEDDING_DIMENSION
    text_embedding_feature_names = [f'text_embedding_dim_{i}' for i in range(bert_embedding_dim)]

    # Filter `model_features` to get only visual feature column names
    visual_feature_cols_for_model = [col for col in model_features if col not in text_embedding_feature_names]
    # Ensure these visual feature columns exist in pdf_df_blocks and are not 'text', 'page_num', 'y0_abs', 'y1_abs'
    visual_feature_cols_actual = [col for col in visual_feature_cols_for_model
                                  if col in pdf_df_blocks.columns and col not in ['text', 'page_num', 'y0_abs', 'y1_abs']]

    visual_features = pdf_df_blocks[visual_feature_cols_actual].values

    # Concatenate text embeddings and visual features
    X_inference = np.concatenate([text_embeddings, visual_features], axis=1)

    # Create DataFrame for LGBM inference, ensuring column names match `model_features`
    inference_df = pd.DataFrame(X_inference, columns=model_features).astype(np.float32)

    # Make predictions using the LGBM classifier
    predictions = lgbm_classifier.predict(inference_df)

    classified_blocks_df = pdf_df_blocks.copy() # Work on a copy
    classified_blocks_df['predicted_label'] = [id2label.get(idx, 'Body') for idx in predictions]

    # --- Post-Processing for Title (Round 1A logic) ---
    doc_title = doc_title_fallback
    if not doc_title or "Microsoft Word" in doc_title: # Common fallback if title not in metadata
        title_candidates = classified_blocks_df[(classified_blocks_df['page_num'] == 1) & (classified_blocks_df['predicted_label'] == 'Title')]
        if not title_candidates.empty:
            doc_title = title_candidates.sort_values(by='pos_y').iloc[0]['text'].strip()
        else:
            # Fallback to largest font size text on page 1 if no 'Title' predicted
            page_1_df = classified_blocks_df[classified_blocks_df['page_num'] == 1]
            if not page_1_df.empty:
                doc_title = page_1_df.loc[page_1_df['font_size'].idxmax()]['text'].strip()

    # --- Hierarchy Building (Round 1A logic) ---
    headings_df = classified_blocks_df[classified_blocks_df['predicted_label'].str.contains(r'^H\d')].copy()

    outline = []
    if not headings_df.empty:
        headings_df['level_num'] = headings_df['predicted_label'].str.replace('H', '').astype(int)
        headings_df = headings_df.sort_values(by=['page_num', 'y0_abs']).reset_index(drop=True)

        path = {} # Stores the current parent node for each level
        for _, row in headings_df.iterrows():
            level = row['level_num']
            node = {"level": f"H{level}", "text": row['text'].strip(), "page": int(row['page_num'])}

            parent = None
            # Find the closest parent (heading of a lower level)
            for i in range(level - 1, 0, -1):
                if i in path:
                    parent = path[i]
                    break

            if parent:
                if "children" not in parent: parent["children"] = []
                parent["children"].append(node)
            else:
                outline.append(node) # Top-level heading

            path[level] = node # Set current node as potential parent

            # Clear paths for higher levels as we are moving down or across a level
            levels_to_remove = [k for k in path if k > level]
            for k in levels_to_remove: del path[k]

    return doc_title, outline, classified_blocks_df

# --- 3. Round 1B Specific Functions ---

def get_full_section_content(classified_blocks_df, outline_nodes, doc_path):
    """
    Extracts the full text content for each identified section (H1, H2, H3)
    based on the classified blocks and the hierarchical outline.
    Also stores individual content blocks for sub-section analysis.
    """
    sections_with_content = []

    # Recursive function to traverse the outline and extract content
    def process_node_and_children(node_list):
        for i, node in enumerate(node_list):
            section_title_text = node['text']
            section_page_start = node['page']

            # Find the starting y0_abs of the current section's heading in the classified blocks
            start_block_candidates = classified_blocks_df[
                (classified_blocks_df['text'] == section_title_text) &
                (classified_blocks_df['page_num'] == section_page_start) &
                (classified_blocks_df['predicted_label'] == node['level']) # Ensure it's the predicted heading
            ]

            start_y_abs = start_block_candidates['y0_abs'].min()

            if pd.isna(start_y_abs):
                # Fallback if exact match not found (due to text stripping, etc.)
                # Try partial match or closest block by y0_abs on the page
                start_y_abs = classified_blocks_df[
                    (classified_blocks_df['page_num'] == section_page_start) &
                    (classified_blocks_df['predicted_label'] == node['level'])
                ].sort_values(by='y0_abs')['y0_abs'].min() # Take the first one of that level on page

                if pd.isna(start_y_abs): # If still not found, skip this node
                    continue

            # Determine the end boundary of the current section's content
            end_y_abs = float('inf') # Default to end of document

            # 1. Check for the start of the next sibling heading (same level)
            if i + 1 < len(node_list):
                next_sibling = node_list[i+1]
                next_sibling_y_abs_candidates = classified_blocks_df[
                    (classified_blocks_df['text'] == next_sibling['text']) &
                    (classified_blocks_df['page_num'] == next_sibling['page']) &
                    (classified_blocks_df['predicted_label'] == next_sibling['level'])
                ]['y0_abs']
                if not next_sibling_y_abs_candidates.empty:
                    end_y_abs = min(end_y_abs, next_sibling_y_abs_candidates.min())

            # 2. Check for the start of the first child heading
            if "children" in node and node["children"]:
                first_child = node["children"][0]
                first_child_y_abs_candidates = classified_blocks_df[
                    (classified_blocks_df['text'] == first_child['text']) &
                    (classified_blocks_df['page_num'] == first_child['page']) &
                    (classified_blocks_df['predicted_label'] == first_child['level'])
                ]['y0_abs']
                if not first_child_y_abs_candidates.empty:
                    end_y_abs = min(end_y_abs, first_child_y_abs_candidates.min())

            # Filter blocks that belong to this section's content
            # Content blocks are all blocks after the current heading's y0_abs,
            # up to the calculated end_y_abs, excluding other headings or meta-data.
            current_section_content_blocks = classified_blocks_df[
                (classified_blocks_df['page_num'] >= section_page_start) &
                (classified_blocks_df['y0_abs'] >= start_y_abs) &
                (classified_blocks_df['y0_abs'] < end_y_abs)
            ].sort_values(by=['page_num', 'y0_abs'])

            # Filter out the heading itself and other non-content labels
            content_only_blocks = current_section_content_blocks[
                ~current_section_content_blocks['predicted_label'].isin(['Title', 'H1', 'H2', 'H3', 'Header', 'Footer', 'Footnote'])
            ].copy() # Ensure we're working on a copy to avoid SettingWithCopyWarning

            # --- IMPORTANT FIX: Convert numerical columns to native Python types ---
            for col in ['font_size', 'relative_font_size', 'pos_y', 'space_above', 'word_count', 'page_num', 'y0_abs', 'y1_abs']:
                if col in content_only_blocks.columns:
                    # Convert to float for floating point numbers, int for integers
                    if content_only_blocks[col].dtype == 'float64' or content_only_blocks[col].dtype == 'float32':
                        content_only_blocks[col] = content_only_blocks[col].astype(float)
                    elif content_only_blocks[col].dtype == 'int64' or content_only_blocks[col].dtype == 'int32':
                        content_only_blocks[col] = content_only_blocks[col].astype(int)
            # --- END FIX ---


            full_section_text = " ".join(content_only_blocks['text'].tolist()).strip()

            sections_with_content.append({
                "document": os.path.basename(doc_path),
                "page_num_start": section_page_start,
                "section_title": section_title_text,
                "section_level": node['level'],
                "full_text_content": full_section_text,
                "sub_blocks": content_only_blocks.to_dict(orient='records') # Now these records should have native Python types
            })

            # Recursively process children of the current node
            if "children" in node:
                process_node_and_children(node["children"])

    process_node_and_children(outline_nodes)
    return sections_with_content

def calculate_cosine_similarity(vec1, vec2):
    """Calculates cosine similarity between two numpy vectors."""
    # Handle cases where one or both vectors are zero vectors to avoid NaN/division by zero
    if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
        return 0.0
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def rank_sections_and_extract_subsections(
    all_document_sections_with_content,
    persona_text,
    job_to_be_done_text,
    min_similarity_threshold # New parameter
):
    """
    Ranks all collected sections from all documents based on the persona and job-to-be-done.
    Filters sections by a minimum similarity threshold and then extracts the most relevant
    sentence ('refined text') for sub-section analysis.
    """
    # Define weights (adjust these values as needed)
    persona_weight = 0.05
    job_weight = 0.95 # Job to be done gets more weight

    # Generate separate embeddings
    persona_embedding = get_text_embedding(persona_text)
    job_embedding = get_text_embedding(job_to_be_done_text)

    # Combine embeddings with weighted average
    # Ensure embeddings are not None and handle cases if one is empty
    if persona_embedding is None:
        persona_embedding = np.zeros(EMBEDDING_DIMENSION, dtype=np.float32)
    if job_embedding is None:
        job_embedding = np.zeros(EMBEDDING_DIMENSION, dtype=np.float32)

    # If both are None (unlikely with dummy data but good practice)
    if np.array_equal(persona_embedding, np.zeros(EMBEDDING_DIMENSION, dtype=np.float32)) and \
       np.array_equal(job_embedding, np.zeros(EMBEDDING_DIMENSION, dtype=np.float32)):
        weighted_query_embedding = np.zeros(EMBEDDING_DIMENSION, dtype=np.float32) # Use EMBEDDING_DIMENSION
    elif np.array_equal(persona_embedding, np.zeros(EMBEDDING_DIMENSION, dtype=np.float32)):
        weighted_query_embedding = job_embedding
    elif np.array_equal(job_embedding, np.zeros(EMBEDDING_DIMENSION, dtype=np.float32)):
        weighted_query_embedding = persona_embedding
    else:
        weighted_query_embedding = (persona_weight * persona_embedding) + \
                                   (job_weight * job_embedding)

    query_embedding = weighted_query_embedding
    sections_with_scores = []

    # Calculate similarity for each section
    for section in all_document_sections_with_content:
        section_text = section['full_text_content']
        section_embedding = get_text_embedding(section_text)

        similarity_score = float(calculate_cosine_similarity(query_embedding, section_embedding)) # Convert to native float

        sections_with_scores.append({
            "document": section['document'],
            "page_num_start": section['page_num_start'],
            "section_title": section['section_title'],
            "section_level": section['section_level'],
            "similarity_score": similarity_score,
            "sub_blocks": section['sub_blocks']
        })

    # Sort all sections across all documents by similarity score (descending)
    sections_with_scores.sort(key=lambda x: x['similarity_score'], reverse=True)

    # --- NEW: Filter sections by similarity threshold ---
    sections_to_process = [
        s for s in sections_with_scores if s['similarity_score'] >= min_similarity_threshold
    ]

    final_extracted_sections = []
    all_subsection_analysis = []

    # Assign importance_rank and extract refined text for filtered sections
    for rank, section_info in enumerate(sections_to_process): # Iterate over the filtered list
        importance_rank = rank + 1

        # Populate extracted_sections output
        final_extracted_sections.append({
            "document": section_info['document'],
            "page_number": section_info['page_num_start'],
            "section_title": section_info['section_title'],
            "importance_rank": importance_rank,
            "similarity_score": section_info['similarity_score'] # Include similarity score for verification
        })

        # --- Sub-section Analysis: Extract Refined Text ---
        # Combine all text from sub_blocks to make it easier to sent_tokenize
        full_section_content_for_sentences = " ".join([b['text'] for b in section_info['sub_blocks']]).strip()

        refined_text_entry = None
        if full_section_content_for_sentences:
            best_sentence = ""
            max_sentence_similarity = -1.0
            best_sentence_page = section_info['page_num_start'] # Default page

            sentences = sent_tokenize(full_section_content_for_sentences)

            for sentence in sentences:
                if not sentence.strip(): continue
                sentence_embedding = get_text_embedding(sentence)
                sentence_similarity = calculate_cosine_similarity(query_embedding, sentence_embedding)

                if sentence_similarity > max_sentence_similarity:
                    max_sentence_similarity = sentence_similarity
                    best_sentence = sentence

            if best_sentence:
                # Try to find the original page number of the best sentence
                for block in section_info['sub_blocks']:
                    if best_sentence in block['text'] or (len(best_sentence) > 20 and best_sentence[:20] in block['text'][:20]): # Simple check for full or start match
                        best_sentence_page = block['page_num']
                        break

                refined_text_entry = {
                    "document": section_info['document'],
                    "refined_text": best_sentence,
                    "page_number": best_sentence_page,
                    "refined_text_similarity": float(max_sentence_similarity) # Convert to native float
                }
                all_subsection_analysis.append(refined_text_entry)

    return final_extracted_sections, all_subsection_analysis

def main_round1b_processing(input_dir, output_dir, persona, job_to_be_done):
    """
    Orchestrates the entire Round 1B processing pipeline:
    1. Processes each PDF using Round 1A logic to get outlines and classified blocks.
    2. Extracts full section content for relevance ranking.
    3. Ranks all sections across all documents based on persona/job.
    4. Extracts refined text for sub-section analysis.
    5. Generates the final Round 1B JSON output.
    """
    os.makedirs(output_dir, exist_ok=True)

    all_docs_sections_for_ranking = []
    all_document_metadata = []

    pdf_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".pdf")]

    if not pdf_files:
        print(f"No PDF files found in '{input_dir}'. Please ensure PDFs are in this directory.")
        return

    print(f"Starting Round 1B processing for {len(pdf_files)} PDF(s)...")

    for pdf_file_name in pdf_files:
        pdf_path = os.path.join(input_dir, pdf_file_name)
        print(f"\n--- Processing '{pdf_file_name}' (Round 1A stage) ---")

        # 1. Extract raw blocks and features
        pdf_df_blocks = extract_blocks_and_features(pdf_path)

        # Get metadata title from PDF itself as initial fallback
        doc_metadata_title = ""
        try:
            temp_doc = fitz.open(pdf_path)
            doc_metadata_title = temp_doc.metadata.get('title', "")
            temp_doc.close()
        except Exception as e:
            print(f"Warning: Could not read metadata title for {pdf_file_name}: {e}")

        # 2. Classify blocks and build Round 1A outline
        doc_title_r1a, outline_r1a, classified_blocks_df = classify_blocks_and_build_outline(pdf_df_blocks, doc_metadata_title)

        # 3. Extract full content for sections based on the outline (New for R1B)
        sections_for_current_doc = get_full_section_content(classified_blocks_df, outline_r1a, pdf_path)
        all_docs_sections_for_ranking.extend(sections_for_current_doc)

        # Store metadata and R1A outline for the final output
        all_document_metadata.append({
            "document_name": pdf_file_name,
            "document_title_from_r1a": doc_title_r1a,
            "outline_from_r1a": outline_r1a
        })

    # 4. Rank all sections across all documents and extract refined text (Core Round 1B logic)
    print("\n--- Ranking sections and extracting sub-sections across all documents ---")
    final_extracted_sections, all_subsection_analysis = rank_sections_and_extract_subsections(
        all_docs_sections_for_ranking, persona, job_to_be_done, MIN_SIMILARITY_THRESHOLD # Pass the new threshold
    )

    # 5. Generate Final Round 1B JSON Output
    output_data = {
        "metadata": {
            "input_documents": [m["document_name"] for m in all_document_metadata],
            "persona": persona,
            "job_to_be_done": job_to_be_done,
            "processing_timestamp": datetime.datetime.now().isoformat(),
            "document_details_from_r1a": all_document_metadata # Includes R1A outlines
        },
        "extracted_sections": final_extracted_sections,
        "sub_section_analysis": all_subsection_analysis
    }

    output_filename = f"round1b_output_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_path = os.path.join(output_dir, output_filename)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)

    print(f"\n--- Round 1B processing complete. Output saved to: {output_path} ---")

if __name__ == '__main__':
    # --- Dummy File Generation for Testing (Remove/Replace for Production) ---
    # This block ensures the script can run out-of-the-box for testing even
    # if Round 1A models or input PDFs are not yet available.

    # Create dummy model files if they don't exist
    if not os.path.exists(MODEL_FILE):
        print(f"Warning: Dummy '{MODEL_FILE}' not found. Creating a dummy for testing purposes.")
        from sklearn.ensemble import RandomForestClassifier
        # Dummy data matching BERT's default hidden size + visual features
        bert_dummy_dim = 128 # Bert-tiny's hidden size is 128
        dummy_visual_features_count = 6 # font_size, is_bold, relative_font_size, pos_y, space_above, word_count
        X_dummy = np.random.rand(100, bert_dummy_dim + dummy_visual_features_count).astype(np.float32)
        y_dummy = np.random.randint(0, 5, 100) # 5 dummy classes (Title, H1, H2, H3, Body)
        dummy_classifier = RandomForestClassifier(random_state=42)
        dummy_classifier.fit(X_dummy, y_dummy)
        joblib.dump(dummy_classifier, MODEL_FILE)

        dummy_features_list = [f'text_embedding_dim_{i}' for i in range(bert_dummy_dim)] + \
                              ['font_size', 'is_bold', 'relative_font_size', 'pos_y', 'space_above', 'word_count']
        with open(FEATURES_FILE, 'w') as f:
            json.dump(dummy_features_list, f)

        dummy_label_map = {"id2label": {0: "Title", 1: "H1", 2: "H2", 3: "H3", 4: "Body"}}
        with open(LABEL_MAP_FILE, 'w') as f:
            json.dump(dummy_label_map, f)

    # Create dummy input PDF files if the directory is empty
    os.makedirs(INPUT_DIR, exist_ok=True)
    if not os.listdir(INPUT_DIR):
        print(f"Warning: No PDF files found in '{INPUT_DIR}'. Creating dummy PDFs for testing.")

        dummy_doc_1 = fitz.open()
        page = dummy_doc_1.new_page()
        page.insert_text((50, 50), "Dummy Company Annual Report 2023", fontname="helv", fontsize=24, color=(0,0,0))
        page.insert_text((50, 100), "1. Introduction to Our Performance", fontname="helv", fontsize=18, color=(0,0,0))
        page.insert_text((50, 130), "This report provides an overview of our financial performance. We saw significant revenue growth this year. Our R&D investments increased by 15% contributing to new product development. Market positioning remains strong.", fontname="helv", fontsize=12, color=(0,0,0))
        page.insert_text((50, 200), "1.1. Financial Highlights", fontname="helv", fontsize=14, color=(0,0,0))
        page.insert_text((50, 230), "Key financial figures demonstrate strong revenue trends. Operating income grew by 10%.", fontname="helv", fontsize=12, color=(0,0,0))
        page.insert_text((50, 280), "2. Research and Development", fontname="helv", fontsize=18, color=(0,0,0))
        page.insert_text((50, 310), "Our commitment to innovation is reflected in our substantial R&D investments. We funded several breakthrough projects.", fontname="helv", fontsize=12, color=(0,0,0))
        dummy_doc_1.save(os.path.join(INPUT_DIR, "dummy_report_A.pdf"))
        dummy_doc_1.close()

        dummy_doc_2 = fitz.open()
        page_2 = dummy_doc_2.new_page()
        page_2.insert_text((50, 50), "Competitor Insights Report 2023", fontname="helv", fontsize=24, color=(0,0,0))
        page_2.insert_text((50, 100), "Chapter 1: Market Overview", fontname="helv", fontsize=18, color=(0,0,0))
        page_2.insert_text((50, 130), "An analysis of the overall market trends. Competitor B showed moderate revenue growth but reduced R&D spending.", fontname="helv", fontsize=12, color=(0,0,0))
        page_2.insert_text((50, 200), "Chapter 2: Competitive Positioning", fontname="helv", fontsize=18, color=(0,0,0))
        page_2.insert_text((50, 230), "Evaluating different market positioning strategies. Competitor C focused on aggressive marketing.", fontname="helv", fontsize=12, color=(0,0,0))
        dummy_doc_2.save(os.path.join(INPUT_DIR, "dummy_report_B.pdf"))
        dummy_doc_2.close()

    # --- Run the main Round 1B processing ---
    main_round1b_processing(INPUT_DIR, OUTPUT_DIR, PERSONA_DEFINITION, JOB_TO_BE_DONE)