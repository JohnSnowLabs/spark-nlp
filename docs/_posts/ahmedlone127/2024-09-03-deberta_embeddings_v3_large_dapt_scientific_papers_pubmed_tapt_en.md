---
layout: model
title: English Deberta Embeddings model (from domenicrosati)
author: John Snow Labs
name: deberta_embeddings_v3_large_dapt_scientific_papers_pubmed_tapt
date: 2024-09-03
tags: [deberta, open_source, deberta_embeddings, debertav2formaskedlm, en, onnx]
task: Embeddings
language: en
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
engine: onnx
annotator: DeBertaEmbeddings
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained DebertaEmbeddings model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `deberta-v3-large-dapt-scientific-papers-pubmed-tapt` is a English model originally trained by `domenicrosati`.

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/deberta_embeddings_v3_large_dapt_scientific_papers_pubmed_tapt_en_5.5.0_3.0_1725331508803.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/deberta_embeddings_v3_large_dapt_scientific_papers_pubmed_tapt_en_5.5.0_3.0_1725331508803.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

embeddings = DeBertaEmbeddings.pretrained("deberta_embeddings_v3_large_dapt_scientific_papers_pubmed_tapt","en") \
    .setInputCols(["document", "token"]) \
    .setOutputCol("embeddings") \
    .setCaseSensitive(True)

pipeline = Pipeline(stages=[documentAssembler, tokenizer, embeddings])

data = spark.createDataFrame([["I love Spark NLP"]]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val documentAssembler = new DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document")

val tokenizer = new Tokenizer()
    .setInputCols("document")
    .setOutputCol("token")

val embeddings = DeBertaEmbeddings.pretrained("deberta_embeddings_v3_large_dapt_scientific_papers_pubmed_tapt","en")
    .setInputCols(Array("document", "token"))
    .setOutputCol("embeddings")
    .setCaseSensitive(true)

val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, embeddings))

val data = Seq("I love Spark NLP").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|deberta_embeddings_v3_large_dapt_scientific_papers_pubmed_tapt|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[deberta]|
|Language:|en|
|Size:|1.6 GB|

## References

References

https://huggingface.co/domenicrosati/deberta-v3-large-dapt-scientific-papers-pubmed-tapt