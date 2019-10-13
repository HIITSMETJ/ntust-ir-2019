import numpy as np
import collections 


# # Readfile

text_file = open('./doc_list.txt', "r")
docs = text_file.read().splitlines()

text_file = open('./query_list.txt', "r")
queries = text_file.read().splitlines()

doc_list=[]
for doc in docs:
    f=open('./Document/'+ doc)
    content = f.read().split()[5:]
    content = [x for x in content if x != '-1']
    doc_list.append(content)

qry_list=[]
for qry in queries:
    f=open('./Query/'+ qry)
    content = f.read().split()    
    content = [x for x in content if x != '-1']
    qry_list.append(content)


# # Lexicon

def creat_lexicon(doc_list):
    flattened = [val for sublist in doc_list for val in sublist]
    all_words=list(set(flattened))
    lexicon=dict(zip(all_words,list(range(len(all_words)))))
    return lexicon


# # Term Frequency

def get_tf(lexicon, file_list):
    
    tf=np.zeros((len(lexicon),len(file_list)))
    for j in range(len(file_list)): 
        content=file_list[j]
        count=dict(collections.Counter(content)) 
        for word in count:
            if word in lexicon:
                i=lexicon[word]
                tf[i][j]=count[word]
    return tf
 
 
# # Term Frequency Factor

def get_doc_len(doc_list):
    doc_len=np.array([len(j) for j in doc_list])
    return doc_len

def get_Fij(tf, doc_len, k1=1.2, b=0.75):
    avg_len=doc_len.mean()
    Fij=np.zeros(tf.shape)
    for j in range(tf.shape[1]):
        len_dj=doc_len[j]
        for i in range(tf.shape[0]):
            Fij[i][j]=(k1+1)*tf[i][j]/(k1*(1-b+b*len_dj/avg_len)+tf[i][j])
    return Fij

def get_Fiq(tf, k3=1.2):
    Fiq=np.zeros(tf.shape)
    for q in range(tf.shape[1]):
        for i in range(tf.shape[0]):
            Fiq[i][q]=(k3+1)*tf[i][q]/(k3+tf[i][q])
    return Fiq


# # Inverse Document Frequency

def get_idf(lexicon, file_list):
    
    df=np.zeros(len(lexicon))
    for j in range(len(file_list)): 
        appear=np.zeros(len(lexicon))
        content=file_list[j]
        count=dict(collections.Counter(content)) 
        for word in count:
            if word in lexicon:
                i=lexicon[word]
                appear[i]=1
        df=np.add(df,appear)
    
    idf = np.log((len(file_list)-df+0.5)/(df+0.5))
    return idf


# # Similarity

def BM25_sim(doc_list,qry_list,lexicon,Fij,Fiq,idf):
    sim=np.zeros((len(qry_list),len(doc_list)))
    for q in range(len(qry_list)):
        for j in range(len(doc_list)):
            intersection=list(set(qry_list[q]) & set(doc_list[j]))
            for word in intersection:
                i=lexicon[word]
                sim[q][j]+=Fij[i][j]*Fiq[i][q]*idf[i]
    return sim


# # Ranking & Output result

# > k = [1.2, 2], b = [0,1]
# 
# * (k1, k3, b) = (1.2, 1.2, 0.75), score = 0.50876
# * (k1, k3, b) = (2.0, 2.0, 0.75), score = 0.53374
# * (k1, k3, b) = (2.0, 2.0, 1.00), score = 0.52032

(k1, k3, b) = (2.0, 2.0, 1.00)
lexicon=creat_lexicon(doc_list)
tfij=get_tf(lexicon, doc_list)
tfiq=get_tf(lexicon, qry_list)
doc_len=get_doc_len(doc_list)
Fij=get_Fij(tfij, doc_len, k1, b)
Fiq=get_Fiq(tfiq, k3)
idf=get_idf(lexicon, doc_list)
sim=BM25_sim(doc_list,qry_list,lexicon,Fij,Fiq,idf)

fname = "./result.txt"
f = open(fname, 'w')
f.write("Query,RetrievedDocuments\n")  

for q in range(len(qry_list)):
    f.write(queries[q] + ",")        
    rank = np.argsort(-sim[q])
    for j in rank:
        f.write(docs[j]+" ")
    f.write("\n")
f.close()
