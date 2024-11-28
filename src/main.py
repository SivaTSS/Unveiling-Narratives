import os
import json
import re
import argparse
import torch
from transformers import pipeline
from collections import defaultdict
import yaml

# Define a set of stopwords
STOPWORDS = set([
    'a', 'an', 'the', 'and', 'but', 'or', 'as', 'at', 'by', 'for', 'in', 'of',
    'on', 'to', 'up', 'with', 'that', 'this', 'is', 'my', 'your', 'he', 'she',
    'it', 'we', 'they', 'you', 'me', 'him', 'her', 'them', 'us', 'our', 'their',
    'his', 'hers', 'its', 'be', 'was', 'were', 'are', 'am', 'do', 'does', 'did',
    'have', 'has', 'had', 'not', 'no', 'yes', 'i', 'mr', 'mrs', 'miss', 'ms',
    'dr', 'sir', 'madam', 'lord', 'lady'
])

def load_book(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def split_into_sentences(text):
    # Simple sentence tokenizer using regex
    sentences = re.split(r'(?<=[.!?])\s+', text)
    print(f"Total sentences: {len(sentences)}")
    return sentences

def extract_mentions(sentences, characters):
    character_contexts = defaultdict(list)
    characters_lower = [char.lower() for char in characters]
    for sentence in sentences:
        sentence_lower = sentence.lower()
        for char, char_lower in zip(characters, characters_lower):
            pattern = r'\b' + re.escape(char_lower) + r'\b'
            if re.search(pattern, sentence_lower):
                character_contexts[char].append(sentence.strip())
    return character_contexts

def count_character_mentions(character_contexts):
    character_counts = {char: len(contexts) for char, contexts in character_contexts.items()}
    return character_counts

def select_top_characters(character_counts, top_n=25):
    sorted_characters = sorted(character_counts.items(), key=lambda x: x[1], reverse=True)
    top_characters = [char for char, count in sorted_characters[:top_n]]
    print(f"Top {top_n} characters selected.")
    return top_characters

def aggregate_contexts(character_contexts, max_sentences=100):
    aggregated = {}
    for char, contexts in character_contexts.items():
        if len(contexts) > max_sentences:
            contexts = contexts[:max_sentences]
        aggregated[char] = ' '.join(contexts)
    return aggregated

def classify_roles_hierarchical(aggregated_contexts, hierarchical_roles, classifier, max_length=1024):
    character_roles = {}
    main_roles = list(hierarchical_roles.keys())
    for char, context in aggregated_contexts.items():
        context = context[:max_length]
        try:
            main_result = classifier(context, main_roles, multi_label=False)
            top_main_role = main_result['labels'][0]
            sub_roles = hierarchical_roles.get(top_main_role, {})
            if sub_roles:
                sub_role_labels = list(sub_roles.keys())
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

def classify_traits(aggregated_contexts, traits, classifier, max_length=1024, top_k=4):
    character_traits = {}
    for char, context in aggregated_contexts.items():
        context = context[:max_length]
        character_traits[char] = {}
        for trait_category, trait_dict in traits.items():
            trait_labels = list(trait_dict.keys())
            try:
                trait_result = classifier(context, trait_labels, multi_label=True)
                # Select top_k traits based on scores
                sorted_traits = sorted(
                    zip(trait_result['labels'], trait_result['scores']),
                    key=lambda x: x[1],
                    reverse=True
                )
                selected_traits = [label for label, score in sorted_traits[:top_k]]
                if not selected_traits:
                    selected_traits.append("None")
                character_traits[char][trait_category] = selected_traits
            except Exception as e:
                print(f"Error classifying traits for {char} in category {trait_category}: {e}")
                character_traits[char][trait_category] = ["Unknown"]
    return character_traits

def load_character_list(file_path, min_length=3):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    person_entities = data.get("Person", [])
    character_names = [entry['word'].strip() for entry in person_entities if entry.get('entity') == "Person"]
    normalized_names = {}
    prefixes = ['Mr.', 'Mrs.', 'Miss', 'Ms.', 'Dr.', 'Sir', 'Madam', 'Lord', 'Lady', 'Mr', 'Mrs', 'Ms', 'Miss']
    for name in character_names:
        # Strip leading/trailing whitespace
        name = name.strip()
        # Remove empty names
        if not name:
            continue
        # Remove names shorter than min_length
        if len(name) < min_length:
            continue
        # Remove names that are in stopwords (case-insensitive)
        if name.lower() in STOPWORDS:
            continue
        # Remove names that contain non-alphabetic characters only
        if not any(c.isalpha() for c in name):
            continue
        # Remove names that are just prefixes like 'Mr', 'Mrs', etc.
        if name.lower() in STOPWORDS:
            continue
        # Normalize name by removing common prefixes
        for prefix in prefixes:
            if name.startswith(prefix):
                name = name[len(prefix):].strip()
                break
        # Remove any leading titles like 'Mr', 'Mrs', etc.
        words = name.split()
        if words and words[0].lower() in STOPWORDS:
            name = ' '.join(words[1:])
        # Remove names shorter than min_length after removing prefixes
        if len(name) < min_length:
            continue
        # Remove names that are in stopwords (again after removing prefixes)
        if name.lower() in STOPWORDS:
            continue
        # Remove names with disallowed characters
        if not all(c.isalpha() or c.isspace() or c == '-' or c == "'" for c in name):
            continue
        # Ensure the name starts with an uppercase letter
        if not name[0].isupper():
            continue
        # Lowercase for key to avoid duplicates
        key = name.lower()
        if key not in normalized_names:
            normalized_names[key] = name
    unique_characters = list(normalized_names.values())
    print(f"Unique characters found: {len(unique_characters)}")
    return unique_characters

def process_book(book_path, character_path, output_path, roles_config, traits_config):
    print("Loading book...")
    text = load_book(book_path)

    print("Splitting into sentences...")
    sentences = split_into_sentences(text)

    print("Loading characters...")
    characters = load_character_list(character_path)

    print("Extracting mentions...")
    extracted_contexts = extract_mentions(sentences, characters)

    print("Counting character mentions...")
    character_counts = count_character_mentions(extracted_contexts)

    print("Selecting top characters...")
    top_characters = select_top_characters(character_counts, top_n=25)

    # Filter the contexts to include only top characters
    top_character_contexts = {char: extracted_contexts[char] for char in top_characters}

    print("Aggregating contexts...")
    aggregated_contexts = aggregate_contexts(top_character_contexts, max_sentences=100)

    print("Initializing classifier...")
    classifier = pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
        device=0 if torch.cuda.is_available() else -1
    )

    if torch.cuda.is_available():
        print("Using GPU for classification.")
    else:
        print("Using CPU for classification.")

    print("Classifying roles...")
    character_roles = classify_roles_hierarchical(aggregated_contexts, roles_config, classifier)

    print("Classifying traits...")
    character_traits = classify_traits(aggregated_contexts, traits_config, classifier)

    print("Combining roles and traits...")
    character_info = {
        char: {
            "Roles": character_roles.get(char, {"Main Role": "Unknown", "Sub Role": "Unknown"}),
            "Traits": character_traits.get(char, {})
        }
        for char in top_characters
    }

    print(f"Saving results to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(character_info, f, indent=4)
    print(f"Character roles saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Character Role and Trait Classification")
    parser.add_argument('--book_path', type=str, required=True, help='Path to the book text file.')
    parser.add_argument('--character_path', type=str, required=True, help='Path to the character JSON file.')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the output JSON.')
    parser.add_argument('--roles_config', type=str, default='config/roles.yaml', help='Path to roles YAML config.')
    parser.add_argument('--traits_config', type=str, default='config/traits.yaml', help='Path to traits YAML config.')

    args = parser.parse_args()

    print("Loading configuration files...")
    with open(args.roles_config, 'r', encoding='utf-8') as file:
        hierarchical_roles = yaml.safe_load(file)

    with open(args.traits_config, 'r', encoding='utf-8') as file:
        traits = yaml.safe_load(file)

    process_book(args.book_path, args.character_path, args.output_path, hierarchical_roles, traits)

if __name__ == "__main__":
    main()
