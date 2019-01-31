import pandas as pd
import _pickle as pickle
import matplotlib.pyplot as plt
import numpy as np


def main():
    # Train dataset path
    filename = 'flickr_8k_train_dataset.txt'
    # Loading data
    df = pd.read_csv(filename, delimiter='\t', encoding='utf8')
    # Getting sample number
    nb_samples = df.shape[0]
    iter = df.iterrows()
    bow = {}
    nbwords = 0
    for i in range(nb_samples):
        x = iter.__next__()
        cap_words = x[1][1].split() # split caption into words
        cap_wordsl = [w.lower() for w in cap_words] # remove capital letters
        nbwords += len(cap_wordsl)
        for w in cap_wordsl:
            if w in bow:
                bow[w] = bow[w]+1
            else:
                bow[w] = 1
    bown = sorted([(value, key) for (key, value) in bow.items()], reverse=True)
    nbkeep = 1000 # 100 is needed for fast processing
    outfile = 'Caption_Embeddings.p'
    [listwords, embeddings] = pickle.load( open(outfile, "rb"))
    embeddings_new = np.zeros((nbkeep,102))
    listwords_new = []
    for i in range(nbkeep):
        listwords_new.append(bown[i][1])
        embeddings_new[i, :] = embeddings[listwords.index(bown[i][1]), :]
        embeddings_new[i, :] /= np.linalg.norm(embeddings_new[i, :]) # Normalization
    listwords = listwords_new
    embeddings = embeddings_new
    outfile = "Caption_Embeddings_"+str(nbkeep)+".p"
    with open(outfile, "wb" ) as pickle_f:
        pickle.dump( [listwords, embeddings], pickle_f)
        freqnc = np.cumsum([float(w[0]) / nbwords * 100.0 for w in bown])
        x_axis = [str(bown[i][1]) for i in range(100)]
        plt.figure(dpi=300)
        plt.xticks(rotation=90, fontsize=3)
        plt.ylabel('Word Frequency')
        plt.bar(x_axis, freqnc[0:100])
        plt.show()
        print("number of kept words=" + str(nbkeep) + " - ratio=" + str(freqnc[nbkeep - 1]) + " %")


if __name__ == '__main__':
    main()