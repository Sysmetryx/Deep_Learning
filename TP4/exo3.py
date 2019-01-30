import pandas as pd
import numpy as np
import _pickle as pickle
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
mpl.use('TKAgg')

def part1():
    # Part One :
    filename = 'flickr_8k_train_dataset.txt'
    df = pd.read_csv(filename, delimiter='\t', encoding='utf8')
    nb_samples = df.shape[0]
    iter = df.iterrows()
    allwords = []
    for i in range(nb_samples):
        x = iter.__next__()
        cap_words = x[1][1].split()  # split caption into words
        cap_wordsl = [w.lower() for w in cap_words]  # remove capital letters
        allwords.extend(cap_wordsl)
    unique = list(set(allwords))  # List of different words in captions
    GLOVE_MODEL = "glove.6B.100d.txt"
    fglove = open(GLOVE_MODEL, "r", encoding='utf8')
    cpt = 0
    listwords = []
    listembeddings = []
    for line in fglove:
        row = line.strip().split()
        word = row[0]
        if (word in unique or word == 'unk'):
            listwords.append(word)
            embedding = np.array(row[1:]).astype(dtype="float32")
            listembeddings.append(embedding)
            cpt += 1
            print("word: " + word + " embedded " + str(cpt))
    fglove.close()
    nbwords = len(listembeddings)
    tembedding = len(listembeddings[0])
    print("Number of words=" + str(
        len(listembeddings)) + " Embedding size=" + str(tembedding))
    embeddings = np.zeros((len(listembeddings) + 2, tembedding + 2))
    for i in range(nbwords):
        embeddings[i, 0:tembedding] = listembeddings[i]
    listwords.append('<start>')
    embeddings[7001, 100] = 1
    listwords.append('<end>')
    embeddings[7002, 101] = 1
    outfile = 'Caption_Embeddings.p'
    with open(outfile, "wb") as pickle_f:
        pickle.dump([listwords, embeddings], pickle_f)

def part2():
    # Part Two :
    outfile = 'Caption_Embeddings.p'
    [listwords, embeddings] = pickle.load(open(outfile, "rb"))
    print("embeddings: " + str(embeddings.shape))
    for i in range(embeddings.shape[0]):
        embeddings[i, :] /= np.linalg.norm(embeddings[i, :])
    kmeans = KMeans(n_clusters=10, n_jobs=16, max_iter=1000,
                    init='random').fit(embeddings)
    clustersID = kmeans.labels_
    clusters = kmeans.cluster_centers_
    indclusters = []
    for i in range(10):
        norm = np.linalg.norm((clusters[i] - embeddings), axis=1)
        inorms = np.argsort(norm)
        indclusters += [inorms[:]]
        print("Cluster " + str(i) + " =" + listwords[indclusters[i][0]])
        for j in range(1, 21):
            print(" mot: " + listwords[indclusters[i][j]])
    tsne = TSNE(n_components=2, perplexity=30, verbose=2, init='pca',
                early_exaggeration=24)
    points2D = tsne.fit_transform(embeddings)
    # initializing variable as emprty array of dimensions (10, 2)
    pointsclusters = [[0] * 2] * 10
    pointsclusters = np.array(pointsclusters)
    for i in range(10):
        pointsclusters[i, :] = points2D[int(indclusters[i][0])]
    cmap = cm.tab10
    plt.figure(figsize=(3.841, 7.195), dpi=100)
    plt.set_cmap(cmap)
    plt.subplots_adjust(hspace=0.4)
    plt.scatter(points2D[:, 0], points2D[:, 1], c=clustersID, s=3,
                edgecolors='none', cmap=cmap, alpha=1.0)
    plt.scatter(pointsclusters[:, 0], pointsclusters[:, 1], c=range(10),
                marker='+', s=1000, edgecolors='none', cmap=cmap, alpha=1.0)

    plt.colorbar(ticks=range(10))
    plt.show()

def main():
    part1()
    part2()

if __name__ == '__main__':
    main()






















