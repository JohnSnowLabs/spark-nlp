---
layout: model
title: Ukrainian DistilBERT Embeddings (from Geotrend)
author: John Snow Labs
name: distilbert_embeddings_distilbert_base_uk_cased
date: 2022-04-12
tags: [distilbert, embeddings, uk, open_source]
task: Embeddings
language: uk
edition: Spark NLP 3.4.2
spark_version: 3.0
supported: true
annotator: DistilBertEmbeddings
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained DistilBERT Embeddings model, uploaded to Hugging Face, adapted and imported into Spark NLP. `distilbert-base-uk-cased` is a Ukrainian model orginally trained by `Geotrend`.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/distilbert_embeddings_distilbert_base_uk_cased_uk_3.4.2_3.0_1649783949701.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/distilbert_embeddings_distilbert_base_uk_cased_uk_3.4.2_3.0_1649783949701.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

embeddings = DistilBertEmbeddings.pretrained("distilbert_embeddings_distilbert_base_uk_cased","uk") \
.setInputCols(["document", "token"]) \
.setOutputCol("embeddings")

pipeline = Pipeline(stages=[documentAssembler, tokenizer, embeddings])

data = spark.createDataFrame([["Я люблю Spark NLP"]]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val documentAssembler = new DocumentAssembler() 
.setInputCol("text") 
.setOutputCol("document")

val tokenizer = new Tokenizer() 
.setInputCols(Array("document"))
.setOutputCol("token")

val embeddings = DistilBertEmbeddings.pretrained("distilbert_embeddings_distilbert_base_uk_cased","uk") 
.setInputCols(Array("document", "token")) 
.setOutputCol("embeddings")

val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, embeddings))

val data = Seq("Я люблю Spark NLP").toDF("text")

val result = pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("uk.embed.distilbert_base_cased").predict("""Я люблю Spark NLP""")
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|distilbert_embeddings_distilbert_base_uk_cased|
|Compatibility:|Spark NLP 3.4.2+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[bert]|
|Language:|uk|
|Size:|195.2 MB|
|Case sensitive:|true|

## References

- https://huggingface.co/Geotrend/distilbert-base-uk-cased
- https://www.aclweb.org/anthology/2020.sustainlp-1.16.pdf
- https://github.com/Geotrend-research/smaller-transformers