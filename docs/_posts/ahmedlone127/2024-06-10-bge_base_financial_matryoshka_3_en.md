---
layout: model
title: English bge_base_financial_matryoshka_3 BGEEmbeddings from Sailesh9999
author: John Snow Labs
name: bge_base_financial_matryoshka_3
date: 2024-06-10
tags: [en, open_source, onnx, embeddings, bge]
task: Embeddings
language: en
edition: Spark NLP 5.4.0
spark_version: 3.0
supported: true
engine: onnx
annotator: BGEEmbeddings
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BGEEmbeddings model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`bge_base_financial_matryoshka_3` is a English model originally trained by Sailesh9999.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bge_base_financial_matryoshka_3_en_5.4.0_3.0_1718061341209.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bge_base_financial_matryoshka_3_en_5.4.0_3.0_1718061341209.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
 
documentAssembler = DocumentAssembler() \
      .setInputCol("text") \
      .setOutputCol("document")

embeddings = BGEEmbeddings.pretrained("bge_base_financial_matryoshka_3","en") \
      .setInputCols(["document"]) \
      .setOutputCol("embeddings")       
        
pipeline = Pipeline().setStages([documentAssembler, embeddings])
data = spark.createDataFrame([["I love spark-nlp"]]).toDF("text")
pipelineModel = pipeline.fit(data)
pipelineDF = pipelineModel.transform(data)

```
```scala

val documentAssembler = new DocumentAssembler() 
    .setInputCol("text") 
    .setOutputCol("document")
    

val embeddings = BGEEmbeddings.pretrained("bge_base_financial_matryoshka_3","en") 
    .setInputCols(Array("document")) 
    .setOutputCol("embeddings")

val pipeline = new Pipeline().setStages(Array(documentAssembler, embeddings))
val data = Seq("I love spark-nlp).toDS.toDF("text")
val pipelineModel = pipeline.fit(data)
val pipelineDF = pipelineModel.transform(data)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bge_base_financial_matryoshka_3|
|Compatibility:|Spark NLP 5.4.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document]|
|Output Labels:|[bge]|
|Language:|en|
|Size:|387.1 MB|

## References

https://huggingface.co/Sailesh9999/bge-base-financial-matryoshka_3