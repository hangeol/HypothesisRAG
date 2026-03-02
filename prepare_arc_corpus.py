import os
import json
from tqdm import tqdm

def prepare_arc_corpus():
    input_file = '/root/HypothesisRAG/data/ARC-V1-Feb2018-2/ARC_Corpus.txt'
    output_dir = '/root/HypothesisRAG/MIRAGE/MedRAG/corpus/arc_corpus/chunk'
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Reading from: {input_file}")
    print(f"Writing to: {output_dir}")
    
    chunk_size = 1000000 # 1 Million lines per file
    chunk_idx = 0
    line_idx = 0
    current_f = None
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            for i, line in tqdm(enumerate(f), desc="Processing ARC Corpus"):
                line = line.strip()
                if not line:
                    continue
                
                if line_idx % chunk_size == 0:
                    if current_f is not None:
                        current_f.close()
                    # Open new chunk file
                    chunk_filename = os.path.join(output_dir, f'arc_chunk_{chunk_idx:03d}.jsonl')
                    current_f = open(chunk_filename, 'w', encoding='utf-8')
                    chunk_idx += 1
                
                # Each jsonl entry must have id, title, content according to MedRAG spec
                # Since ARC has no titles, we use an empty title and the line as content
                doc = {
                    "id": f"arc_doc_{i}",
                    "title": "",
                    "content": line
                }
                
                current_f.write(json.dumps(doc) + '\n')
                line_idx += 1
                
    finally:
        if current_f is not None:
            current_f.close()
            
    print(f"Finished processing {line_idx} documents into {chunk_idx} chunk files.")
    
if __name__ == "__main__":
    prepare_arc_corpus()
