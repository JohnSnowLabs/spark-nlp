---
layout: model
title: English Legal Word2Vec Embeddings (Lemmatized)
author: John Snow Labs
name: word2vec_osf_lemmatized_legal
date: 2022-12-20
tags: [word2vec, en, english, embeddings, open_source]
task: Embeddings
language: en
nav_key: models
edition: Spark NLP 4.2.5
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Legal Word Embeddings lookup annotator that maps tokens to vectors. Trained on legal text after lemmatization.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/word2vec_osf_lemmatized_legal_en_4.2.5_3.0_1671534245861.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/word2vec_osf_lemmatized_legal_en_4.2.5_3.0_1671534245861.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
 
model = WordEmbeddingsModel.pretrained("word2vec_osf_lemmatized_legal","en")\
	            .setInputCols(["document","token"])\
	            .setOutputCol("word_embeddings")

```
```scala

val model = WordEmbeddingsModel.pretrained("word2vec_osf_lemmatized_legal","en")
	                .setInputCols("document","token")
	                .setOutputCol("word_embeddings")

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|word2vec_osf_lemmatized_legal|
|Type:|embeddings|
|Compatibility:|Spark NLP 4.2.5+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[embeddings]|
|Language:|en|
|Size:|53.9 MB|
|Case sensitive:|false|
|Dimension:|100|

## References

https://osf.io/
