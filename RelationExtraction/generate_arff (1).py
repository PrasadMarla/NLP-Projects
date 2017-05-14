import nltk
import json
nltk.download('punkt')
import os
import csv
import sys
from DependencyParse import *
from nltk.parse import stanford
os.environ['STANFORD_PARSER'] = '/home/prasad/Desktop/NLP ASSG3/stanford-parser-full-2016-10-31/stanford-parser.jar'
os.environ['STANFORD_MODELS'] = '/home/prasad/Desktop/NLP ASSG3/stanford-parser-full-2016-10-31/stanford-parser-3.7.0-models.jar'
sys.path.append('/home/prasad/Desktop/NLP ASSG3/stanford-parser-full-2016-10-31/stanford-parser.jar')
from nltk.parse.stanford import StanfordParser
parser=StanfordParser(model_path="englishPCFG.ser.gz")
from nltk.tag.stanford import StanfordPOSTagger
englishTag = StanfordPOSTagger(model_filename="/home/prasad/Desktop/NLP ASSG3/english-bidirectional-distsim.tagger",path_to_jar="/home/prasad/Desktop/NLP ASSG3/stanford-postagger.jar")
import pandas as pd
from  nltk import word_tokenize
from nltk.tokenize import TreebankWordTokenizer

TEST_DATA_PATH = "test.tsv"
TRAIN_DATA_PATH = "train.tsv"


def parse_data(train_data, test_data):
    """
    Input: path to the data file
    Output: (1) a list of tuples, one for each instance of the data, and
            (2) a list of all unique tokens in the data

    Parses the data file to extract all instances of the data as tuples of the form:
    (person, institution, judgment, full snippet, intermediate text)
    where the intermediate text is all tokens that occur between the first occurrence of
    the person and the first occurrence of the institution.

    Also extracts a list of all tokens that appear in the intermediate text for the
    purpose of creating feature vectors.
    """
    all_tokens = []
    data = []
    for fp in [train_data, test_data]:
        with open(fp) as f:
            for line in f:
                institution, person, snippet, intermediate_text, judgment = line.split("\t")
                judgment = judgment.strip()

                # Build up a list of unique tokens that occur in the intermediate text
                # This is needed to create BOW feature vectors
                #tokens = word_tokenize(intermediate_text)
                #tokens = TreebankWordTokenizer().tokenize(intermediate_text)
                #print(tokens)
                tokens = intermediate_text.split()
                for t in tokens:
                    t = t.lower()
                    if t not in all_tokens:
                        all_tokens.append(t)
                data.append((person, institution, judgment, snippet, intermediate_text))
    return data, all_tokens


def create_feature_vectors(data, all_tokens):
    """
    Input: (1) The parsed data from parse_data()
             (2) a list of all unique tokens found in the intermediate text
    Output: A list of lists representing the feature vectors for each data instance

    Creates feature vectors from the parsed data file. These features include
    bag of words features representing the number of occurrences of each
    token in the intermediate text (text that comes between the first occurrence
    of the person and the first occurrence of the institution).
    This is also where any additional user-defined features can be added.
    """
    feature_vectors = []
    lenAll = len(all_tokens)
    #lenAll = 28404
    print(lenAll)
    i = 0;
    for instance in data:
        # BOW features
        # Gets the number of occurrences of each token
        # in the intermediate text
        feature_vector = [0]*lenAll
        intermediate_text = instance[4]
        tokens = intermediate_text.split()
        #tokens = word_tokenize(intermediate_text)
        #tokens = TreebankWordTokenizer().tokenize(intermediate_text)
        for token in tokens:
            #if all_tokens.__contains__(token.lower):
            index = all_tokens.index(token.lower())
            feature_vector[index] += 1

        ### ADD ADDITIONAL FEATURES HERE ###
        #res,all_tokens = findbetweenWords(instance[3],instance[1],all_tokens)
        #print(len(all_tokens))
        #feature_vector.append(res[0])
        #feature_vector.append(res[1])
        # Class label
        judgment = instance[2]
        feature_vector.append(judgment)

        feature_vectors.append(feature_vector)
    return feature_vectors,all_tokens


def depTagFeature(inpTag):
    result = [0]*len(tags)
    for tag in inpTag:
        result[tags.index(tag.split(":")[0])]+=1
    return result


"""
def findbetweenWords(tokens,institues,all_tokens):
    #print(tokens)
    #print(institues)
    temp = tokens.partition(institues)
    #ind = tokens.index(institues)
    #print(ind)
    res = [0,0]


    if len(temp)==3:
        splitp = temp[0].split()
        if len(splitp) !=0:
            if splitp[len(splitp)-1] not in all_tokens:
                all_tokens.append(splitp[len(splitp)-1].lower())
            res[0] = (all_tokens.index(splitp[len(splitp)-1].lower()))

        splite = temp[2].split()
        if len(splite) != 0:
            if splite[0].lower() not in all_tokens:
                all_tokens.append(splite[0].lower())
            res[1] = (all_tokens.index(splite[0].lower()))
            #print(res)
    return  res,all_tokens

def getNumberofVerbs():
    sents = []
    trees = []
    verbCount = 0;
    i = 0
    with open("test_out.conllx", 'r') as fin:
        #sentenceTokens = []
        for line in fin:
            if line in ['\n','\r\n']:
                sents.append(verbCount)
                verbCount = 0
            else:
                line = line.strip()
                line = line.split('\t')
                if line[3] in ['VERB']:
                    print(line[1])
                    verbCount = verbCount + 1
    print(sents)


"""

tags = ['ref', 'nummod', 'mark', 'iobj', 'dobj', 'ROOT', 'csubjpass', 'aux', 'conj', 'nmod', 'punct', 'compound', 'nsubj', 'expl', 'case', 'parataxis', 'auxpass', 'advcl', 'acl', 'cc', 'det', 'xcomp', 'dep', 'nsubjpass', 'cop', 'root', 'advmod', 'csubj', 'ccomp', 'neg', 'mwe', 'amod', 'appos']
# method handles dependency features
def getDependencyFeatures(train_data, test_data):
    all_tokens = []
    data = []
    #f2 = open('Output.txt', 'w')
    #featurewriter = csv.writer(f2, delimiter='\t')
    i = 0
    resMap = {}
    for fp in [train_data, test_data]:
        f2 = open("Output"+str(i)+".txt", 'w')
        featurewriter = csv.writer(f2, delimiter='\t')
        i = i + 1
        with open(fp) as f:
            for line in f:
                institution, person, snippet, intermediate_text, judgment = line.split("\t")
                judgment = judgment.strip()
                uniText = institution.split(" ")[0]
                if institution.split(" ")[0] in ["the","of"]:
                    uniText = institution.split(" ")[1]
                tags = englishTag.tag(tokens=institution.split(" "))
                for tag in tags:
                    if tag[1] in ["NNP"]:
                        uniText = tag[0]
                out,path,nSub,deps,tagMap = generateDependencyPath(snippet,person.split(" ")[0], uniText)
                resMap.update(tagMap)
                outList = []
                vCount = calculateVerbCount(out)
                pLen = lenPath(out)
                isSub = isSubjectCorrect(nSub)
                depTags = depTagFeature(deps)
                print(depTags)
                print([vCount,pLen,isSub,judgment]+depTags)
                featurewriter.writerow([vCount,pLen,isSub,judgment]+depTags)
                print(out)
    json.dump(resMap,open("tags.txt",'w'))


# Verb count in shortest path feature. 
def calculateVerbCount(inpList):
    count = 0
    for intList in inpList:
        print(intList)
        if intList[2].startswith("V"):
            count = count+1;
    return  count

# Shortest path length
def lenPath(inpList):
    return  len(inpList)

# Subject Relation exist feature
def isSubjectCorrect(nSub):
    if nSub.startswith("nsubj"):
        return True
    return False

def generate_arff_file(feature_vectors, all_tokens, out_path):
    """
    Input: (1) A list of all feature vectors for the data
             (2) A list of all unique tokens that occurred in the intermediate text
             (3) The name and path of the ARFF file to be output
    Output: an ARFF file output to the location specified in out_path

    Converts a list of feature vectors to an ARFF file for use with Weka.
    """
    with open(out_path, 'w') as f:
        # Header info
        f.write("@RELATION institutions\n")
        for i in range(len(all_tokens)):
            f.write("@ATTRIBUTE token_{} INTEGER\n".format(i))

        ### SPECIFY ADDITIONAL FEATURES HERE ###
        # For example: f.write("@ATTRIBUTE custom_1 REAL\n")
        f.write("@ATTRIBUTE m1+1 INTEGER\n")
        f.write("@ATTRIBUTE m1-1 INTEGER\n")
        # Classes
        f.write("@ATTRIBUTE class {yes,no}\n")

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


def generate_brown_arff_file(feature_vectors, all_tokens, out_path, feature_size):
    """
    Input: (1) A list of all feature vectors for the data
             (2) A list of all unique tokens that occurred in the intermediate text
             (3) The name and path of the ARFF file to be output
    Output: an ARFF file output to the location specified in out_path

    Converts a list of feature vectors to an ARFF file for use with Weka.
    """
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



if __name__ == "__main__":
    #ependencyParse("helo")
    getDependencyFeatures(TRAIN_DATA_PATH, TEST_DATA_PATH)
    #generateDependencyCSV()
    #data, all_tokens = parse_data(TRAIN_DATA_PATH, TEST_DATA_PATH)
    #feature_vectors,all_tokens = create_feature_vectors(data, all_tokens)
    #print(len(all_tokens))
    #generate_arff_file(feature_vectors[:6000], all_tokens, "train.arff")
    #generate_arff_file(feature_vectors[6000:], all_tokens, "test.arff")
