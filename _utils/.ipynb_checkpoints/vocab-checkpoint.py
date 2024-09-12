import paddle, os, time

from collections import Counter
from itertools import chain

import jieba

def sort_and_write_words(all_words, file_path):
    words = list(chain(*all_words))
    words_vocab = Counter(words).most_common()
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write('[UNK]\n[PAD]\n')
        for word, num in words_vocab:
            if num < 5:
                continue
            f.write(word + '\n')

(root, directory, files), = list(os.walk('readonly'))

all_words = []

for file in files:
    with open(os.path.join(root, file), 'r', encoding = 'utf-8') as f:
        for line in f:
            if file in ['train.txt', 'dev.txt']:
                text, label = line.strip().split('\t')
            elif file == 'test.txt':
                text = line.strip()
            else:
                continue
            words = jieba.lcut(text)
            words = [word for word in words if word.strip() != '']
            all_words.append(words)

sort_and_write_words(all_words, '.\\train\\vocab.txt')