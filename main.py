import numpy as np
import os
from sklearn.cluster import MiniBatchKMeans as kmeans


def load_word2vec(dir):
    word2vec = {}
    for path in os.listdir(dir):
        iword2vec = {}
        # Load the word2vec features.
        with open(os.path.join(dir, path), 'r', encoding='utf-8') as fin:
            if path == 'vectors0.txt':
                next(fin)  # Skip first line
            for line in fin:
                items = line.strip().split(' ')
                if len(items) < 10:
                    continue
                word = items[0]
                vect = np.array([float(i) for i in items[1:] if len(i) > 1])
                iword2vec[word] = vect

        word2vec.update(iword2vec)

    return word2vec


def get_furthest_word(words, word2vect):
    vectlist = []
    for word in words:
        if word not in word2vect:
            return word  # Return unknown word
        vectlist.append(word2vect[word] / np.linalg.norm(word2vect[word]))  # Normalize
    mean = np.array(vectlist).mean(axis=0)
    mean = mean / np.linalg.norm(mean)

    dists = [np.linalg.norm(v - mean) for v in vectlist]
    return words[np.argmax(dists)]


def cluster_vects(word2vect):
    clusters = kmeans(n_clusters=25, max_iter=10, batch_size=200,
                      n_init=1, init_size=2000)
    X = np.array([i.T for i in word2vect.values()])
    y = list(word2vect.keys())

    print('Fitting k-means, may take some time...')
    clusters.fit(X)
    print('Done.')

    return {word: label for word, label in zip(y, clusters.labels_)}


def words_in_cluster(word, word_to_label):
    label = word_to_label[word]
    return [key for key, val in word_to_label.items() if val == label]


def main():
    print('Loading knowledge from Wikipedia...should take 10-20 seconds')
    word2vec = load_word2vec('vectors')
    print('Type several words separated by spaces. The more words you enter, the better I can guess.')

    while True:
        words = input('-> ').lower().split(' ')
        print(f'I think {get_furthest_word(words, word2vec)} doesn\'t belong in this list!\n')


if __name__ == '__main__':
    main()