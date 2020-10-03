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
