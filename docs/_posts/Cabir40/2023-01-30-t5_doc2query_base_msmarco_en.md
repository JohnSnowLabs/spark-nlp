---
layout: model
title: English T5ForConditionalGeneration Base Cased model (from macavaney)
author: John Snow Labs
name: t5_doc2query_base_msmarco
date: 2023-01-30
tags: [en, open_source, t5, tensorflow]
task: Text Generation
language: en
nav_key: models
edition: Spark NLP 4.3.0
spark_version: 3.0
supported: true
engine: tensorflow
annotator: T5Transformer
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained T5ForConditionalGeneration model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `doc2query-t5-base-msmarco` is a English model originally trained by `macavaney`.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/t5_doc2query_base_msmarco_en_4.3.0_3.0_1675101186297.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/t5_doc2query_base_msmarco_en_4.3.0_3.0_1675101186297.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = DocumentAssembler() \
    .setInputCols("text") \
    .setOutputCols("document")

t5 = T5Transformer.pretrained("t5_doc2query_base_msmarco","en") \
    .setInputCols("document") \
    .setOutputCol("answers")
    
pipeline = Pipeline(stages=[documentAssembler, t5])

data = spark.createDataFrame([["PUT YOUR STRING HERE"]]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val documentAssembler = new DocumentAssembler() 
      .setInputCols("text")
      .setOutputCols("document")
       
val t5 = T5Transformer.pretrained("t5_doc2query_base_msmarco","en") 
    .setInputCols("document")
    .setOutputCol("answers")
   
val pipeline = new Pipeline().setStages(Array(documentAssembler, t5))

val data = Seq("PUT YOUR STRING HERE").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|t5_doc2query_base_msmarco|
|Compatibility:|Spark NLP 4.3.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[documents]|
|Output Labels:|[t5]|
|Language:|en|
|Size:|474.3 MB|

## References

- https://huggingface.co/macavaney/doc2query-t5-base-msmarco
- https://git.uwaterloo.ca/jimmylin/doc2query-data/raw/master/T5-passage/t5-base.zip
- https://github.com/terrierteam/pyterrier_doc2query
- https://cs.uwaterloo.ca/~jimmylin/publications/Nogueira_Lin_2019_docTTTTTquery-v2.pdf
- https://arxiv.org/abs/2007.14271