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
INPUT_DIR = "sample_input2" # Directory for input PDFs and the input JSON
OUTPUT_DIR = "output_results33" # Directory for the output JSON
BERT_CHECKPOINT = "my_finetuned_minilm_model" # Keep the existing BERT model

# --- Similarity Thresholds and Window Sizes ---
MIN_SIMILARITY_THRESHOLD = 0.25 # Minimum similarity for sections to be considered
SUBSECTION_BLOCK_WINDOW_SIZE = 1 # Number of blocks to include before/after the most relevant block for refined_text

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
    min_similarity_threshold,
    subsection_block_window_size
):
    """
    Ranks all collected sections from all documents based on the persona and job-to-be-done.
    Filters sections by a minimum similarity threshold and then extracts the most relevant
    block(s) for sub-section analysis to form "refined text."
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

    # --- Filter sections by similarity threshold ---
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
            "importance_rank": importance_rank
            # Removed 'similarity_score' to match desired output format
        })

        # --- Sub-section Analysis: Extract Refined Text (now a group of concurrent blocks) ---
        best_block_info = None
        max_block_similarity = -1.0

        for idx, sub_block in enumerate(section_info['sub_blocks']):
            block_text = sub_block['text']
            if not block_text.strip(): continue

            block_embedding = get_text_embedding(block_text)
            block_similarity = calculate_cosine_similarity(query_embedding, block_embedding)

            if block_similarity > max_block_similarity:
                max_block_similarity = block_similarity
                best_block_info = (sub_block, idx)

        refined_text_entry = None
        if best_block_info:
            best_block, best_block_idx = best_block_info

            # Determine the window of blocks to include for "concurrently occurring sentences"
            start_idx = max(0, best_block_idx - subsection_block_window_size)
            end_idx = min(len(section_info['sub_blocks']), best_block_idx + subsection_block_window_size + 1)

            selected_block_texts = []
            selected_block_pages = []

            for i in range(start_idx, end_idx):
                block_to_add = section_info['sub_blocks'][i]
                selected_block_texts.append(block_to_add['text'])
                selected_block_pages.append(block_to_add['page_num'])

            combined_refined_text = " ".join(selected_block_texts).strip()

            # The page number will be the page of the most relevant block
            refined_text_page = best_block['page_num']

            refined_text_entry = {
                "document": section_info['document'],
                "refined_text": combined_refined_text,
                "page_number": refined_text_page
                # Removed 'refined_text_similarity' to match desired output format
            }
            all_subsection_analysis.append(refined_text_entry)

    return final_extracted_sections, all_subsection_analysis

def main_round1b_processing(input_dir, output_dir, input_json_path):
    """
    Orchestrates the entire Round 1B processing pipeline:
    1. Reads input parameters and document list from input JSON.
    2. Processes each specified PDF using Round 1A logic to get outlines and classified blocks.
    3. Extracts full section content for relevance ranking.
    4. Ranks all sections across all documents based on persona/job.
    5. Extracts refined text (concurrent blocks) for sub-section analysis.
    6. Generates the final Round 1B JSON output.
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"Attempting to load input JSON from: {input_json_path}")
    # Read input JSON
    try:
        with open(input_json_path, 'r', encoding='utf-8') as f:
            input_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input JSON file not found at '{input_json_path}'. Please ensure it exists.")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{input_json_path}'. Please check file format.")
        return


    persona = input_data['persona']['role']
    job_to_be_done = input_data['job_to_be_done']['task']
    pdf_files_info = input_data['documents']
    pdf_filenames = [doc['filename'] for doc in pdf_files_info] # Extract just filenames

    all_docs_sections_for_ranking = []
    all_document_metadata = []

    if not pdf_filenames:
        print(f"No PDF files specified in '{input_json_path}'. Please ensure 'documents' list is populated.")
        return

    print(f"Starting Round 1B processing for {len(pdf_filenames)} PDF(s) specified in '{input_json_path}'...")

    for pdf_file_name in pdf_filenames:
        pdf_path = os.path.join(input_dir, pdf_file_name)
        
        if not os.path.exists(pdf_path):
            print(f"Warning: PDF file '{pdf_file_name}' not found at '{pdf_path}'. Skipping this document.")
            continue # Skip to the next file if not found

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
        all_docs_sections_for_ranking, persona, job_to_be_done, MIN_SIMILARITY_THRESHOLD, SUBSECTION_BLOCK_WINDOW_SIZE
    )

    # 5. Generate Final Round 1B JSON Output
    output_data = {
        "metadata": {
            "input_documents": pdf_filenames, # Use the list extracted from input JSON
            "persona": persona,
            "job_to_be_done": job_to_be_done,
            "processing_timestamp": datetime.datetime.now().isoformat(),
            "document_details_from_r1a": all_document_metadata # Includes R1A outlines
        },
        "extracted_sections": final_extracted_sections,
        "subsection_analysis": all_subsection_analysis
    }

    output_filename = f"round1b_output_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_path = os.path.join(output_dir, output_filename)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)

    print(f"\n--- Round 1B processing complete. Output saved to: {output_path} ---")

if __name__ == '__main__':
    os.makedirs(INPUT_DIR, exist_ok=True)
    
    # Define the expected input JSON file name and path
    INPUT_JSON_FILEPATH = "input.json"
    

    # --- Run the main Round 1B processing with the actual input JSON path ---
    main_round1b_processing(INPUT_DIR, OUTPUT_DIR, INPUT_JSON_FILEPATH)