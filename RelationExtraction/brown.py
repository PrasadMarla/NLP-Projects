import csv
import re

def generate_brown_arff_file(feature_vectors, all_tokens, out_path, feature_size):
    
    with open(out_path, 'w') as f:
        # Header info
        f.write("@RELATION institutions\n")
        for i in range(feature_size):
            f.write("@ATTRIBUTE token_brown_{} integer\n".format(i))

        ### SPECIFY ADDITIONAL FEATURES HERE ###
        # For example: f.write("@ATTRIBUTE custom_1 REAL\n")

        # Classes
        f.write("@ATTRIBUTE class_brown {yes,no}\n")

        # Data instances
        f.write("\n@DATA\n")
        for fv in feature_vectors:
            features = []
            for i in range(len(fv)):
                value = fv[i]
                if value != 0:
                    features.append("{} {}".format(i, value))
            entry = ",".join(features)
            f.write("{" + entry + "}\n")


def trim_cluster_ids(clust_ids, prlen):
    ids = []
    for id in clust_ids:
        if len(id) > prlen:
            id = id[:prlen]
        if id not in ids:
            ids.append(id)
    return ids

def parse_brown_clusters(filename):
    f = open(filename)
    all_id = []
    word_map = dict()
    for line in f:
        id, word, count = line.split("\t")
        id = id.strip()
        word = word.strip()
        word_map[word] = id
        if id not in all_id:
            all_id.append(id)

    return all_id, word_map


def create_trim_brown_features(data, cluster_ids, word_map, prefix_len):
    feature_vectors = []
    for instance in data:
        feature_vector = [0] * len(cluster_ids)
        intermediate_text = instance[4]
        tokens = intermediate_text.split()
        for token in tokens:
            c_id = word_map.get(token)
            if len(c_id) > prefix_len:
                c_id = c_id[:prefix_len]
            index = cluster_ids.index(c_id)
            feature_vector[index] += 1

            judgment = instance[2]
        feature_vector.append(judgment)

        feature_vectors.append(feature_vector)
    return feature_vectors


def create_brown_features(data, cluster_ids, word_map):
    feature_vectors = []
    for instance in data:
        feature_vector = [0] * len(cluster_ids)
        intermediate_text = instance[4]
        # intermediate_text = intermediate_text.strip()
        tokens = intermediate_text.split()
        for token in tokens:
            print(token)
            c_id = word_map.get(token)
            print(c_id)
            index = cluster_ids.index(c_id)
            feature_vector[index] += 1
        judgment = instance[2]
        feature_vector.append(judgment)
        feature_vectors.append(feature_vector)
    return feature_vectors

