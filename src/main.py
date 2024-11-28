import os
import json
import re
import argparse
from transformers import pipeline
from tqdm import tqdm
from collections import defaultdict
import yaml

def load_book(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def split_into_chunks(text, chunk_size=1000, overlap=200):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = ' '.join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    print("Number of chunks:", len(chunks))
    return chunks

def extract_mentions(chunks, characters):
    character_contexts = defaultdict(list)
    characters_lower = [char.lower() for char in characters]
    for chunk in chunks:
        for char, char_lower in zip(characters, characters_lower):
            pattern = r'\b' + re.escape(char_lower) + r'\b'
            matches = re.finditer(pattern, chunk.lower())
            for match in matches:
                sentence_start = chunk.rfind('.', 0, match.start()) + 1
                sentence_end = max(
                    chunk.rfind('.', 0, match.start()),
                    chunk.rfind('!', 0, match.start()),
                    chunk.rfind('?', 0, match.start())
                )
                if sentence_start == -1:
                    sentence_start = 0
                sentence = chunk[sentence_start:sentence_end].strip()
                if sentence:
                    character_contexts[char].append(sentence)
    return character_contexts

def aggregate_contexts(character_contexts):
    aggregated = {char: ' '.join(contexts) for char, contexts in character_contexts.items()}
    return aggregated

def classify_roles_hierarchical(aggregated_contexts, hierarchical_roles, classifier, max_length=1024):
    character_roles = {}
    main_roles = list(hierarchical_roles.keys())
    for char, context in aggregated_contexts.items():
        if len(context) > max_length:
            context = context[:max_length]
        try:
            main_result = classifier(context, main_roles, multi_label=False)
            top_main_role = main_result['labels'][0]
            sub_roles = hierarchical_roles.get(top_main_role, [])
            if sub_roles:
                sub_role_labels = [sub_role.split(":")[0] for sub_role in sub_roles]
                sub_result = classifier(context, sub_role_labels, multi_label=False)
                top_sub_role = sub_result['labels'][0] if sub_result['labels'] else "Unknown"
            else:
                top_sub_role = "Unknown"
            character_roles[char] = {
                "Main Role": top_main_role,
                "Sub Role": top_sub_role
            }
        except Exception as e:
            print(f"Error classifying {char}: {e}")
            character_roles[char] = {
                "Main Role": "Unknown",
                "Sub Role": "Unknown"
            }
    return character_roles

def classify_traits(aggregated_contexts, traits, classifier, max_length=1024, threshold=0.6):
    character_traits = {}
    for char, context in aggregated_contexts.items():
        if len(context) > max_length:
            context = context[:max_length]
        character_traits[char] = {}
        for trait_category, trait_list in traits.items():
            trait_labels = [trait.split(":")[0] for trait in trait_list]
            try:
                trait_result = classifier(context, trait_labels, multi_label=True)
                selected_traits = [
                    label for label, score in zip(trait_result['labels'], trait_result['scores']) 
                    if score >= threshold
                ]
                if not selected_traits:
                    selected_traits.append("None")
                character_traits[char][trait_category] = selected_traits
            except Exception as e:
                print(f"Error classifying traits for {char} in category {trait_category}: {e}")
                character_traits[char][trait_category] = ["Unknown"]
    return character_traits

def load_character_list(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    person_entities = data.get("Person", [])
    character_names = [entry['word'] for entry in person_entities if entry['entity'] == "Person"]
    normalized_names = {}
    for name in character_names:
        key = name.lower()
        if key not in normalized_names:
            normalized_names[key] = name
    unique_characters = list(normalized_names.values())
    print(f"Unique characters found: {len(unique_characters)}")
    return unique_characters

def process_book(book_path, character_path, output_path, roles_config, traits_config):
    text = load_book(book_path)
    chunks = split_into_chunks(text)
    characters = load_character_list(character_path)
    extracted_contexts = extract_mentions(chunks, characters)
    aggregated_contexts = aggregate_contexts(extracted_contexts)
    print("Aggregated Contexts:", aggregated_contexts)
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=0 if torch.cuda.is_available() else -1)
    character_roles = classify_roles_hierarchical(aggregated_contexts, roles_config, classifier)
    character_traits = classify_traits(aggregated_contexts, traits_config, classifier)
    character_info = {
        char: {
            "Roles": character_roles.get(char, {"Main Role": "Unknown", "Sub Role": "Unknown"}),
            "Traits": character_traits.get(char, {})
        }
        for char in characters
    }
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(character_info, f, indent=4)
    print(f"Character roles saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Character Role and Trait Classification")
    parser.add_argument('--book_path', type=str, required=True, help='Path to the book text file.')
    parser.add_argument('--character_path', type=str, required=True, help='Path to the character JSON file.')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the output JSON.')
    parser.add_argument('--roles_config', type=str, default='config/hierarchical_roles.yaml', help='Path to roles YAML config.')
    parser.add_argument('--traits_config', type=str, default='config/traits.yaml', help='Path to traits YAML config.')
    
    args = parser.parse_args()
    
    with open(args.roles_config, 'r', encoding='utf-8') as file:
        hierarchical_roles = yaml.safe_load(file)
    
    with open(args.traits_config, 'r', encoding='utf-8') as file:
        traits = yaml.safe_load(file)
    
    process_book(args.book_path, args.character_path, args.output_path, hierarchical_roles, traits)

if __name__ == "__main__":
    import torch
    main()
