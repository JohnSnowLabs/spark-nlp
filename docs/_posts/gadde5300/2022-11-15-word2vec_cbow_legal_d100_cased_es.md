---
layout: model
title: Spanish CBOW Legal Fast Text Embeddings (Cased, D100)
author: John Snow Labs
name: word2vec_cbow_legal_d100_cased
date: 2022-11-15
tags: [fasttext, es, spanish, embeddings, open_source]
task: Embeddings
language: es
edition: Spark NLP 4.2.1
spark_version: 3.0
supported: true
annotator: WordEmbeddingsModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Word Embeddings lookup annotator that maps tokens to vectors. In CBOW the distributed representations of context (or surrounding words) are combined to predict the word in the middle.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/word2vec_cbow_legal_d100_cased_es_4.2.1_3.0_1668534200139.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
 
model = WordEmbeddingsModel.pretrained("word2vec_cbow_legal_d100_cased","es")\
	            .setInputCols(["document","token"])\
	            .setOutputCol("word_embeddings")

```
```scala

val model = WordEmbeddingsModel.pretrained("word2vec_cbow_legal_d100_cased","es")
	                .setInputCols("document","token")
	                .setOutputCol("word_embeddings")

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|word2vec_cbow_legal_d100_cased|
|Type:|embeddings|
|Compatibility:|Spark NLP 4.2.1+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[embeddings]|
|Language:|es|
|Size:|387.4 MB|
|Case sensitive:|false|
|Dimension:|300|

## References

https://zenodo.org/record/5036147#.Y3Op0XZBxD-