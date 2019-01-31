import time
import numpy as np
import _pickle as pickle


def main():
    bStart = False
    fin = open('fleurs_mal.txt', 'r', encoding='utf8')
    lines = fin.readlines()
    lines2 = []
    text = []
    for line in lines:
        line = line.strip().lower() # Remove blanks and capitals
        if("Charles Baudelaire avait un ami".lower()
           in line and bStart == False):
            print("START")
            bStart = True
        if("End of the Project Gutenberg EBook of Les Fleurs du Mal, "
           "by Charles Baudelaire".lower() in line):
            print("END")
            break
        if bStart == False or len(line) == 0:
            continue
        lines2.append(line)
    fin.close()
    text = " ".join(lines2)
    chars = sorted(set([c for c in text]))
    # chars is a list of a all characters present in text. It is sorted as
    # alphabetical (Special char first, then numbers, then letters)
    nb_chars = len(chars)
    # nb_chars is the number of different characters in the text.
    SEQLEN = 10  # Length of the sequence to predict next char
    STEP = 1  # stride between two subsequent sequences
    input_chars = []
    label_chars = []
    for i in range(0, len(text) - SEQLEN, STEP):
        input_chars.append(text[i:i + SEQLEN])
        label_chars.append(text[i + SEQLEN])
    nbex = len(input_chars)
    # mapping char -> index in dictionary: used for encoding (here)
    char2index = dict((c, i) for i, c in enumerate(chars))
    # mapping char -> index in dictionary:
    # used for decoding, i.e. generation - part c)
    index2char = dict((i, c) for i, c in
                      enumerate(chars))  # mapping index -> char in dictionary
    X = np.zeros((len(input_chars), SEQLEN, nb_chars), dtype=np.bool)
    y = np.zeros((len(input_chars), nb_chars), dtype=np.bool)
    # print(input_chars)
    for i, input_char in enumerate(input_chars):
        for j, ch in enumerate(input_char):
            # Fill X at correct index
            X[i, j, char2index[ch]] = True
        # Fill y at correct index
        y[i, char2index[label_chars[i]]] = True
    ratio_train = 0.8
    nb_train = int(round(len(input_chars) * ratio_train))
    print("nb tot=", len(input_chars), "nb_train=", nb_train)
    X_train = X[0:nb_train, :, :]
    y_train = y[0:nb_train, :]
    X_test = X[nb_train:, :, :]
    y_test = y[nb_train:, :]
    print("X train.shape=", X_train.shape)
    print("y train.shape=", y_train.shape)
    print("X test.shape=", X_test.shape)
    print("y test.shape=", y_test.shape)
    outfile = "Baudelaire_len_" + str(SEQLEN) + ".p"
    with open(outfile, "wb") as pickle_f:
        pickle.dump([index2char, X_train, y_train, X_test, y_test], pickle_f)


if __name__ == '__main__':
    start_time = time.time()
    main()
    print("-------% " + str(time.time() - start_time) + " secondes %-------")
