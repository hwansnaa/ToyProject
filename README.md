# ToyProject
Summary of text using Text Rank Algorothm


### requirements
+ konlpy // 한국어 정보처리를 위한 패키지
+ sikit-learn
+ numpy

### kkma를 이용한 문장분석 [문장 단위로 Slice]
```
def kkmasentences(corpus):
    kkma = Kkma()
    text = ''.join(corpus)
    _sentences = kkma.sentences(text)
    sentences = [x for x in _sentences if len(x) > 10] // 길이가 10 이하인 text는 큰 의미가 없으므로 삭제 ex. 아, 안녕하세요 etc.
    return sentences
```

### Twitter를 이용해 문장을 단어 단위로 Tokenize
```
def tokenize(sentences, stopwords):
    twitter = Twitter()
    nouns = []
    for sentence in sentences:
        nouns.append(''.join([noun for noun in twitter.couns(str(sentences)) if noun not in stopwords and len(noun) > 2])) // 불용어와 길이가 2 이하인 단어들 제거 ex. 아, 아니 etc.
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
