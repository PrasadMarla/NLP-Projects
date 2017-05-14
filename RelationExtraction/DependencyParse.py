import networkx as nx
from pycorenlp import StanfordCoreNLP
from pprint import pprint

nlp = StanfordCoreNLP('http://localhost:{0}'.format(9000))
def get_stanford_annotations(text, port=9000,
                             annotators='tokenize,ssplit,pos,lemma,depparse,parse'):
    output = nlp.annotate(text, properties={
        "timeout": "10000",
        "ssplit.newlineIsSentenceBreak": "two",
        "ssplit.isOneSentence": "true",
        'annotators': annotators,
        'outputFormat': 'json',
        'pos.model':'/home/prasad/Desktop/NLP ASSG3/english-bidirectional-distsim.tagger'
    })
    return output

def generateDependencyPath(document, token1, token2):
    #document = "Gerald Clarke was born in Hemet, California in 1967 to Carol and Gerald Clarke, Sr., his father being born Cahuilla. At the age of 3 his parents divorced and he moved with his siblings and mother to Orange County, and on the weekends he would return to the reservation to spend time with his father. At age 16, he moved to Arkansas with his mother and sister. He attended Ozarka College, where he majored in welding, electrical maintenance, and hydraulics; three necessary components to the artworks Clarke would create as a full time artist."
    print('document: {0}'.format(document))
    annotations = get_stanford_annotations(document, port=9000,
                                           annotators='tokenize,ssplit,pos,lemma,depparse')
    tokens = annotations['sentences'][0]['tokens']
    # Load Stanford CoreNLP's dependency tree into a networkx graph
    edges = []
    tagMap = {}
    dependencies = {}
    nSubpos = ""
    #prinedt(annotations)
    map1 = {}
    for edge in annotations['sentences'][0]['enhancedDependencies']:
        edges.append((edge['governor'], edge['dependent']))
        map1[str(edge['governor']) + "-" + str(edge['dependent'])] = edge['dep']
        map1[str(edge['dependent']) + "-" + str(edge['governor'])] = edge['dep']
        #print(edge['dep'])
        tagMap[edge['dep']] = 1
        if "nsubj" in edge['dep']  and token1 in edge['dependentGloss']:
            nSubpos = edge['dep']
            print("Hello")
            print(nSubpos)
        dependencies[(min(edge['governor'], edge['dependent']),
                      max(edge['governor'], edge['dependent']))] = edge

    graph = nx.Graph(edges)
    #pprint(dependencies)
    # print('edges: {0}'.format(edges))

    # Find the shortest path
    print(map1)
    print(token1)
    print(token2)
    print(tokens)
    #token1.replace("'s","")
    #token2.replace("'s", "")
    for token in tokens:
        if token1 == token['originalText'] or token1 in token['originalText']   : #or token['originalText'] in token1:
            token1_index = token['index']
        elif "'" in token1:
            if token['originalText'] in token1.split("'"):
                token1_index = token['index']
        if token2 == token['originalText'] or token2 in token['originalText']: #or token['originalText'] in token2:
            token2_index = token['index']
        elif "'" in token2:
            print(token2)
            toks = token2.split("'")
            print(toks)
            print(token['originalText'])
            if token['originalText'] in toks:
                token2_index = token['index']

    path = nx.shortest_path(graph, source=token1_index, target=token2_index)
    print('path: {0}'.format(path))
    result = []
    i = 0
    deps1 = []
    for i in range(len(path)-1):
        deps1.append(map1.get(str(path[i])+"-"+str(path[i+1])))
    print(deps1)
    for token_id in path:
        toks = []
        token = tokens[token_id - 1]
        #print(token)
        token_text = token['originalText']
        token_POS =  token['pos']
        result.append([token_id,token_text,token_POS])
        print('Node {0}\ttoken_text: {1} \t pos:{2}'.format(token_id, token_text,token_POS))
    return result,path,nSubpos,deps1,tagMap