---
layout: model
title: Word2Vec Gigaword and Wikipedia - 300 dimensions
author: John Snow Labs
name: word2vec_gigaword_wiki_300
date: 2022-01-03
tags: [word2vec, en, english, wikipedia, word_embeddings, embeddings, open_source]
task: Embeddings
language: en
edition: Spark NLP 3.4.0
spark_version: 3.0
supported: true
annotator: Word2VecModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

We have trained this Word2Vec model by using Gigaword 5th Edition and English Wikipedia Dump of February 2017 over the window size of 5 and 300 dimensions. We used the `Word2VecApproach` annotator that uses the Spark ML Word2Vec behind the scene to train a Word2Vec model.

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/word2vec_gigaword_wiki_300_en_3.4.0_3.0_1641224007056.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
document = DocumentAssembler()\
.setInputCol("text")\
.setOutputCol("document")

token = Tokenizer()\
.setInputCols("document")\
.setOutputCol("token")

norm = Normalizer()\
.setInputCols(["token"])\
.setOutputCol("normalized")\
.setLowercase(True)

stops = StopWordsCleaner.pretrained()\
.setInputCols("normalized")\
.setOutputCol("cleanedToken")

doc2Vec = Word2VecModel.pretrained("word2vec_gigaword_wiki_300", "en")\
.setInputCols("cleanedToken")\
.setOutputCol("sentence_embeddings")
```
```scala
val document = new DocumentAssembler()
.setInputCol("text")
.setOutputCol("document")

val tokenizer = new Tokenizer()
.setInputCols(Array("document"))
.setOutputCol("token")

val norm = new Normalizer()
.setInputCols(Array("token"))
.setOutputCol("normalized")
.setLowercase(true)

val stops = StopWordsCleaner.pretrained()
.setInputCols("normalized")
.setOutputCol("cleanedToken")

val doc2Vec = Word2VecModel.pretrained("word2vec_gigaword_wiki_300", "en")
.setInputCols("cleanedToken")\
.setOutputCol("sentence_embeddings")
```


{:.nlu-block}
```python
import nlu
nlu.load("en.embed.word2vec.gigaword_wiki").predict("""Put your text here.""")
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|word2vec_gigaword_wiki_300|
|Compatibility:|Spark NLP 3.4.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[token]|
|Output Labels:|[embeddings]|
|Language:|en|
|Size:|326.7 MB|

## Data Source

[https://catalog.ldc.upenn.edu/LDC2011T07](https://catalog.ldc.upenn.edu/LDC2011T07)

[https://dumps.wikimedia.org/enwiki/](https://dumps.wikimedia.org/enwiki/)