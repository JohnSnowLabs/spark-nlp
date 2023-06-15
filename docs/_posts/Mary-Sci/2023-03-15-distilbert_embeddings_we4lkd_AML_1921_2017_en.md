---
layout: model
title: English Bert Embeddings model (from matheusvolpon)
author: John Snow Labs
name: distilbert_embeddings_we4lkd_AML_1921_2017
date: 2023-03-15
tags: [open_source, distilbert, distilbert_embeddings, distilbertformaskedlm, en, tensorflow]
task: Embeddings
language: en
edition: Spark NLP 4.3.1
spark_version: 3.0
supported: true
engine: tensorflow
annotator: DistilBertEmbeddings
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained DistilBertForMaskedLM model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `WE4LKD_AML_distilbert_1921_2017` is a English model originally trained by `matheusvolpon`.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/distilbert_embeddings_we4lkd_AML_1921_2017_en_4.3.1_3.0_1678893104164.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/distilbert_embeddings_we4lkd_AML_1921_2017_en_4.3.1_3.0_1678893104164.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

embeddings = DistilBertEmbeddings.pretrained("distilbert_embeddings_we4lkd_AML_1921_2017","en") \
    .setInputCols(["document", "token"]) \
    .setOutputCol("embeddings") \
    .setCaseSensitive(True)

pipeline = Pipeline(stages=[documentAssembler, tokenizer, embeddings])

data = spark.createDataFrame([["I love Spark-NLP"]]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val documentAssembler = new DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document")

val tokenizer = new Tokenizer()
    .setInputCols("document")
    .setOutputCol("token")

val embeddings = DistilBertEmbeddings.pretrained("distilbert_embeddings_we4lkd_AML_1921_2017","en")
    .setInputCols(Array("document", "token"))
    .setOutputCol("embeddings")
    .setCaseSensitive(True)

val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, embeddings))

val data = Seq("I love Spark-NLP").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|distilbert_embeddings_we4lkd_AML_1921_2017|
|Compatibility:|Spark NLP 4.3.1+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[embeddings]|
|Language:|en|
|Size:|249.5 MB|
|Case sensitive:|false|

## References

https://huggingface.co/matheusvolpon/WE4LKD_AML_distilbert_1921_2017
