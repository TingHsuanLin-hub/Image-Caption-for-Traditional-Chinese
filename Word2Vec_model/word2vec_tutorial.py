from gensim.models import word2vec

## load model 
model = word2vec.Word2Vec.load('word2vec_512.model')

## convert word to vector
embedding = model.wv["畢業"]
print(embedding)

## find most similar word vector
similar = model.wv.most_similar(embedding)
print(similar)

## convert to word
word = similar[0][0]
print(word)