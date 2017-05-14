import re
from nltk.tokenize import RegexpTokenizer
import nltk
import csv

tokenizer = RegexpTokenizer(r'\w+')
TEST_DATA_PATH = "test.tsv"
TRAIN_DATA_PATH = "train.tsv"


def parse_data(train_data, test_data):
    data = []
    for fp in [train_data, test_data]:
        with open(fp, encoding="utf8") as f:
            for line in f:
                institution, person, snippet, intermediate_text, judgment = line.split("\t")
                judgment = judgment.strip()
                data.append((intermediate_text, judgment))

    return data


def calResult(trp, fap, trn, fan):
    recall = trp / (trp + fan)
    precision = trp / (trp + fap)
    f1 = 2 * precision * recall / (precision + recall)
    print('Precision :', precision)
    print('Recall :', recall)
    print('F1 Score :', f1)
    return


# word based  regular expression
def word_manual_rule_based_extractor(data):
    tp = 0
    fp = 0
    tn = 0
    fn = 0

    rule_list = [
        re.compile(r'.*educate.*',re.IGNORECASE),
        re.compile(r'.*complet.*', re.IGNORECASE),
        re.compile(r'.*graduat.*', re.IGNORECASE),
        re.compile(r'.*attend.*', re.IGNORECASE),
        re.compile(r'.*student.*', re.IGNORECASE)]

    for line in data:
        sent = line[0]
        labl = line[1]

        if any(pattern.match(sent) for pattern in rule_list):
            if labl == 'yes':
                tp += 1
            else:
                fp += 1
        else:
            if labl == 'no':
                tn += 1
            else:
                fn += 1

    return


# pos based  regular expression
def pos_manual_rule_based_extractor(data):
    trp = 0
    fap = 0
    tn = 0
    fn = 0

    rule_list = [
        re.compile(r'.*VBD.*IN.*'),
        re.compile(r'.*VBD.*PRP.*'),
        re.compile(r'.*VBD.*DT.*'),
        re.compile(r'.*VBD.*VBN.*'),
        re.compile(r'.*PRP.*VBD.*')
    ]

    for line in data:
        sent = line[0]
        labl = line[1]
        sent_t = nltk.word_tokenize(sent)
        tags = nltk.pos_tag(sent_t)
        sent_tag = ' '.join([tag[1] for tag in tags])

        if any(pattern.match(sent_tag) for pattern in rule_list):
            # print(sent, sent_tag)
            # print('\n')
            if labl == 'yes':
                trp += 1
            else:
                fap += 1
        else:
            if labl == 'no':
                tn += 1
            else:
                fn += 1

    calResult(trp, fap, tn, fn)
    return


#pos+word regular expression 
def pos_word_manual_rule_based_extractor(data):
    tp = 0
    fp = 0
    tn = 0
    fn = 0

    word_rule_list = [
        # re.compile(r'.*earn.*',re.IGNORECASE),
        re.compile(r'.*graduat.*', re.IGNORECASE),
        re.compile(r'.*attend.*', re.IGNORECASE),
        re.compile(r'.*complet.*',re.IGNORECASE),
        re.compile(r'.*student.*', re.IGNORECASE)]

    pos_rule_list = [
        re.compile(r'.*VBD.*PRP.*'),
        re.compile(r'.*VBD.*DT.*'),
        re.compile(r'.*VBD.*VBN.*'),
        re.compile(r'.*VBD.*IN.*'),
        re.compile(r'.*PRP.*VBD.*')
    ]

    for line in data:
        sent = line[0]
        labl = line[1]
        sent_t = nltk.word_tokenize(sent)
        tags = nltk.pos_tag(sent_t)
        sent_tag = ' '.join([tag[1] for tag in tags])

        if any(pos_pattern.match(sent_tag) for pos_pattern in pos_rule_list) and any(
                word_pattern.match(sent) for word_pattern in word_rule_list):
            if labl == 'yes':
                tp += 1
            else:
                fp += 1

        else:
            if labl == 'no':
                tn += 1
            else:
                fn += 1

    calResult(tp, fp, tn, fn)
    return


def pos_word():
    case = re.IGNORECASE
    reg_exps_words = [re.compile(r'.*graduate*', case), re.compile(r'.*educat*', case),
                      re.compile(r'.*bach*', case), re.compile(r'.*attend*', case), re.compile(r'.*stud*', case)]
    true_positives, true_negatives, false_positives, false_negatives, data_no = 0, 0, 0, 0, 0
    reg_exps_tags = [re.compile(r'.*VBD.*PRP.*'),
        re.compile(r'.*VBD.*DT.*'),
        re.compile(r'.*VBD.*VBN.*'),
        re.compile(r'.*VBD.*IN.*'),
        re.compile(r'.*PRP.*VBD.*')]

    for fp in [TRAIN_DATA_PATH, TEST_DATA_PATH]:
        data_no += 1
        with open(fp) as f:
            data = csv.reader(f, delimiter='\t')
            for line in data:
                sentence = line[2]
                sentence_tags = ' '.join([t[1] for t in nltk.pos_tag(sentence)])
                # print sentence, sentence_tags
                if any(r.match(sentence_tags) for r in reg_exps_tags) and \
                        any(s.match(sentence) for s in reg_exps_words):
                    if line[4] == "yes":
                        true_positives += 1
                    else:
                        false_positives += 1
                else:
                    if line[4] == "no":
                        true_negatives += 1
                    else:
                        false_negatives += 1

            calResult(true_positives, false_positives, true_negatives, false_negatives)

if __name__ == "__main__":
    data = parse_data(TRAIN_DATA_PATH, TEST_DATA_PATH)
    word_manual_rule_based_extractor(data)
    pos_manual_rule_based_extractor(data)
    pos_word_manual_rule_based_extractor(data)
