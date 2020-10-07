# ToyProject
Summary of text using Text Rank Algorothm


### requirements
+ konlpy # 한국어 정보처리를 위한 패키지
+ sikit-learn
+ numpy

### kkma를 이용한 문장분석 [문장 단위로 Slice]
```
def kkmasentences(corpus):
    kkma = Kkma()
    text = ''.join(corpus)
    _sentences = kkma.sentences(text)
    sentences = [x for x in _sentences if len(x) > 10] # 길이가 10 이하인 text는 큰 의미가 없으므로 삭제 ex. 아, 안녕하세요 etc.
    return sentences
```

### Twitter를 이용해 문장을 단어 단위로 Tokenize
```
def tokenize(sentences, stopwords):
    twitter = Twitter()
    nouns = []
    for sentence in sentences:
        nouns.append(''.join([noun for noun in twitter.nouns(str(sentence)) if noun not in stopwords and len(noun) > 2])) # 불용어와 길이가 2 이하인 단어들 제거 ex. 아, 아니 etc.
    return nouns
```

### TF-IDF를 사용해 DTM 내 각 단어의 중요도를 계산하여 Correlation matrix로 표현
```
def get_CorrMatrix(sentence):
    tfifd = TfidfVectorizer()
    tfidf_mat = tfidf.fit_transform(sentence).toarray()
    graph_sentence = np.dot(tfidf_mat, tfidf_mat.T)
    return graph_sentence
```

### TextRankAlgorithm
```
def get_ranks(graph, d=0.85): # d = damping factor
    A = graph
    matrix_size = A.shape[0]
    for id in range(matrix_size):
        A[id, id] = 0 # diagonal 부분을 0으로
        link_sum = np.sum(A[:,id]) # A[:, id] = A[:][id]
        if link_sum != 0:
            A[:, id] /= link_sum
        A[:, id] *= -d
        A[id, id] = 1
    B = (1-d) * np.ones((matrix_size, 1))
    ranks = np.linalg.solve(A, B) # 연립방정식 Ax = b
    return {idx: r[0] for idx, r in enumerate(ranks)}
```
### Summarize (main.py)
```
def summarize(corpus, stopwords, sent_num=5):
    sentences = kkmasentences(corpus)
    nouns = tokenize(sentences, stopwords)
    sent_graph = get_CorrMatrix(nouns)
    sent_rank_idx = get_ranks(sent_graph)
    sorted_sent_rank_idx = sorted(sent_rank_idx, key=lambda k: sent_rank_idx[k], reverse=True)

    summary = []
    index=[]
    for idx in sorted_sent_rank_idx[:sent_num]:
        index.append(idx)
    index.sort()
    for idx in index:
        summary.append(sentences[idx])
    return summary
```
