import os
import json
import re
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline
import torch
from transformers import RobertaTokenizerFast

def process_ner_predictions(ner_results, text, threshold=0.0):
    # Mapping from entity labels to categories
    entity_mapping = {
        'PER': 'Person',
        'ORG': 'Organization',
        'LOC': 'Location'
    }

    # ignoring 'MISC'
    filtered_results = [
        res for res in ner_results
        if res['score'] >= threshold and res['entity'].replace('B-', '').replace('I-', '') in entity_mapping
    ]

    # sorting predictions
    filtered_results = sorted(filtered_results, key=lambda x: x['start'])
    entities = []
    current_entity = None

    for res in filtered_results:
        label = res['entity'].replace('B-', '').replace('I-', '')
        if label not in entity_mapping:
            continue  
        category = entity_mapping[label]
        start = res['start']
        end = res['end']
        word = text[start:end]
        score = float(res['score'])

        if current_entity is None:
            current_entity = {
                'entity': category,
                'score': score,
                'word': word,
                'start': start,
                'end': end
            }
        else:
            if (start <= current_entity['end'] + 1) and (current_entity['entity'] == category):
                current_entity['end'] = max(current_entity['end'], end)
                current_entity['word'] = text[current_entity['start']:current_entity['end']]
                current_entity['score'] = max(current_entity['score'], score)
            else:
                entities.append(current_entity)
                current_entity = {
                    'entity': category,
                    'score': score,
                    'word': word,
                    'start': start,
                    'end': end
                }

    if current_entity is not None:
        entities.append(current_entity)

    grouped_entities = {'Person': [], 'Organization': [], 'Location': []}
    for entity in entities:
        category = entity['entity']
        grouped_entities[category].append(entity)

    return grouped_entities


def split_text_into_chunks(text, words_per_chunk=256, words_stride=50):
    word_pattern = re.compile(r'\S+')
    matches = list(word_pattern.finditer(text))
    words = [match.group(0) for match in matches]
    word_positions = [(match.start(), match.end()) for match in matches]
    chunks = []
    i = 0
    while i < len(words):
        chunk_word_positions = word_positions[i:i+words_per_chunk]
        if not chunk_word_positions:
            break
        chunk_start_pos = chunk_word_positions[0][0]
        chunk_end_pos = chunk_word_positions[-1][1]
        chunk_text = text[chunk_start_pos:chunk_end_pos]
        chunks.append((chunk_text, chunk_start_pos))
        i += words_per_chunk - words_stride
    return chunks

def main():
    model_path = "ner_model" 
    tokenizer_path = "tokenizer"

    model = AutoModelForTokenClassification.from_pretrained(model_path)
    tokenizer = RobertaTokenizerFast.from_pretrained(tokenizer_path)

    device = 0 if torch.cuda.is_available() else -1
    nlp = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="none", device=device)

    if device >= 0:
        print("Using GPU for inference.")
    else:
        print("Using CPU for inference.")

    input_dir = "../Gutenberg/txt/"  
    output_dir = "ner_results_roberta" 

    os.makedirs(output_dir, exist_ok=True)

    threshold = 0.5

    for filename in os.listdir(input_dir):
        if filename.endswith(".txt"):
            input_path = os.path.join(input_dir, filename)
            output_filename = os.path.splitext(filename)[0] + ".json"
            output_path = os.path.join(output_dir, output_filename)

            with open(input_path, "r", encoding="utf-8") as f:
                text = f.read()

            chunks = split_text_into_chunks(text, words_per_chunk=256, words_stride=50)

            all_predictions = []
            for chunk_text, chunk_start in chunks:
                ner_results = nlp(chunk_text)
                for res in ner_results:
                    res['start'] += chunk_start
                    res['end'] += chunk_start

                all_predictions.extend(ner_results)

            processed_entities = process_ner_predictions(all_predictions, text, threshold)

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(processed_entities, f, ensure_ascii=False, indent=2)

            print(f"Processed '{filename}' and saved results to '{output_filename}'.")

if __name__ == "__main__":
    main()
