import os
import json
from collections import defaultdict

def analyze_metadata(base_path):
    language_file_count = defaultdict(int)
    language_subjects = defaultdict(set)

    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file == "metadata.jsonl":
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            metadata = json.loads(line.strip())
                            languages = metadata.get("languages", [])
                            subjects = metadata.get("subjects", "")
                            
                            # Count each language in the languages array
                            for language in languages:
                                language_file_count[language] += 1
                            
                            # Process subjects
                            if subjects:
                                for subject in subjects.split(","):
                                    for language in languages:
                                        language_subjects[language].add(subject.strip())
                except Exception as e:
                    print(f"Error reading file {file_path}: {e}")

    print("Statistics:")
    print("Number of metadata.jsonl files per language:")
    for language, count in language_file_count.items():
        print(f"{language}: {count}")

    print("\nNumber of different subjects per language:")
    for language, subjects in language_subjects.items():
        print(f"{language}: {len(subjects)}")

if __name__ == "__main__":
    base_path = "/srv/liri_storage/data/yingqiang/projects/spirituality/health/crawled_data/"
    analyze_metadata(base_path)