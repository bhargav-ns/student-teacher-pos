import os
from collections import defaultdict

DATA_PATH = 'data/UD_English-EWT'

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    sentences = []
    sentence = []
    tags = []

    for line in content.strip().split('\n'):
        if line.startswith('#'):  # Skip comments
            continue
        if line == '':
            sentences.append((' '.join(sentence), tags))
            sentence = []
            tags = []
        else:
            parts = line.split('\t')
            sentence.append(parts[1])  # Word
            tags.append(parts[3])  # POS tag

    return sentences

train_data = load_data(os.path.join(DATA_PATH, 'en_ewt-ud-train.conllu'))
dev_data = load_data(os.path.join(DATA_PATH, 'en_ewt-ud-dev.conllu'))
test_data = load_data(os.path.join(DATA_PATH, 'en_ewt-ud-test.conllu'))

word_to_ix = defaultdict(lambda: len(word_to_ix))
tag_to_ix = defaultdict(lambda: len(tag_to_ix))
ix_to_tag = {}

# Creating the vocabularies
for sentence, tags in train_data:
    for word, tag in zip(sentence.split(), tags):
        word_to_ix[word]  # Automatically increments the index if the word is new
        if tag not in tag_to_ix:
            tag_to_ix[tag] = len(tag_to_ix)
            ix_to_tag[tag_to_ix[tag]] = tag