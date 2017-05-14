import pickle
import numpy as np

(dictionary, embeddings) = pickle.load(open('word2vec.model', 'rb'))


def read_data(filename, output_filename):
    rdata = open(filename, 'r')
    contents = rdata.read()
    lines = contents.split('\n')

    file2 = open(output_filename, 'w')

    for line in lines:
        splits_arr = line.split(',')
        diffarr = list()
        outline = ''
        for pair in splits_arr:
            if pair:
                outline += pair
                outline += ' '
                words = pair.strip('"').split(':')
                if words[0] in dictionary:
                    v1 = embeddings[dictionary[words[0]]]
                else:
                    v1 = embeddings[dictionary['UNK']]
                if words[0] in dictionary:
                    v2 = embeddings[dictionary[words[0]]]
                else:
                    v2 = embeddings[dictionary['UNK']]
                dotprod = np.dot(v1, v2)
                cosine = dotprod / (np.sqrt(np.dot(v1, v1))
                                    * np.sqrt(np.dot(v2, v2)))
                diffarr.append(cosine)
        if len(diffarr) != 0:
            outline += splits_arr[diffarr.index(min(diffarr))]
            outline += ' '
            outline += splits_arr[diffarr.index(max(diffarr))]
            file2.write(outline)
            file2.write('\n')
    file2.close()
    return '1'


words = read_data('Testing.txt', 'resultChecker.txt')


			