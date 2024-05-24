---
layout: model
title: English Bert Embeddings model (from Rundstedtz)
author: John Snow Labs
name: distilbert_embeddings_base_uncased_letters_from_jenny
date: 2024-05-24
tags: [open_source, distilbert, distilbert_embeddings, distilbertformaskedlm, en, onnx]
task: Embeddings
language: en
edition: Spark NLP 5.2.4
spark_version: 3.0
supported: true
engine: onnx
annotator: DistilBertEmbeddings
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained DistilBertForMaskedLM model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `distilbert-base-uncased-letters-from-jenny` is a English model originally trained by `Rundstedtz`.

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/distilbert_embeddings_base_uncased_letters_from_jenny_en_5.2.4_3.0_1716562437525.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/distilbert_embeddings_base_uncased_letters_from_jenny_en_5.2.4_3.0_1716562437525.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

embeddings = DistilBertEmbeddings.pretrained("distilbert_embeddings_base_uncased_letters_from_jenny","en") \
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

val embeddings = DistilBertEmbeddings.pretrained("distilbert_embeddings_base_uncased_letters_from_jenny","en")
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
|Model Name:|distilbert_embeddings_base_uncased_letters_from_jenny|
|Compatibility:|Spark NLP 5.2.4+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[distilbert]|
|Language:|en|
|Size:|247.2 MB|

## References

References

https://huggingface.co/Rundstedtz/distilbert-base-uncased-letters-from-jenny