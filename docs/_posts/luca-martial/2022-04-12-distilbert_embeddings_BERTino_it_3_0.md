---
layout: model
title: Italian DistilBERT Embeddings
author: John Snow Labs
name: distilbert_embeddings_BERTino
date: 2022-04-12
tags: [distilbert, embeddings, it, open_source]
task: Embeddings
language: it
edition: Spark NLP 3.4.2
spark_version: 3.0
supported: true
recommended: true
annotator: DistilBertEmbeddings
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained DistilBERT Embeddings model, uploaded to Hugging Face, adapted and imported into Spark NLP. `BERTino` is a Italian model orginally trained by `indigo-ai`.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/distilbert_embeddings_BERTino_it_3.4.2_3.0_1649783809783.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = DocumentAssembler() \
.setInputCol("text") \
.setOutputCol("document")

tokenizer = Tokenizer() \
.setInputCols("document") \
.setOutputCol("token")

embeddings = DistilBertEmbeddings.pretrained("distilbert_embeddings_BERTino","it") \
.setInputCols(["document", "token"]) \
.setOutputCol("embeddings")

pipeline = Pipeline(stages=[documentAssembler, tokenizer, embeddings])

data = spark.createDataFrame([["Adoro Spark NLP"]]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val documentAssembler = new DocumentAssembler() 
.setInputCol("text") 
.setOutputCol("document")

val tokenizer = new Tokenizer() 
.setInputCols(Array("document"))
.setOutputCol("token")

val embeddings = DistilBertEmbeddings.pretrained("distilbert_embeddings_BERTino","it") 
.setInputCols(Array("document", "token")) 
.setOutputCol("embeddings")

val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, embeddings))

val data = Seq("Adoro Spark NLP").toDF("text")

val result = pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("it.embed.BERTino").predict("""Adoro Spark NLP""")
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|distilbert_embeddings_BERTino|
|Compatibility:|Spark NLP 3.4.2+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[bert]|
|Language:|it|
|Size:|253.3 MB|
|Case sensitive:|true|

## References

- https://huggingface.co/indigo-ai/BERTino
- https://indigo.ai/en/
- https://www.corpusitaliano.it/
- https://corpora.dipintra.it/public/run.cgi/corp_info?corpname=itwac_full
- https://universaldependencies.org/treebanks/it_partut/index.html
- https://universaldependencies.org/treebanks/it_isdt/index.html
- https://figshare.com/articles/Learning_multilingual_named_entity_recognition_from_Wikipedia/5462500
