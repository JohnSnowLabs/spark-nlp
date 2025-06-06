---
layout: model
title: English gpl_e5_base_test_1_aba799 E5Embeddings from rithwik-db
author: John Snow Labs
name: gpl_e5_base_test_1_aba799
date: 2024-09-03
tags: [en, open_source, onnx, embeddings, e5]
task: Embeddings
language: en
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
engine: onnx
annotator: E5Embeddings
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained E5Embeddings model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`gpl_e5_base_test_1_aba799` is a English model originally trained by rithwik-db.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/gpl_e5_base_test_1_aba799_en_5.5.0_3.0_1725344159704.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/gpl_e5_base_test_1_aba799_en_5.5.0_3.0_1725344159704.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
 
documentAssembler = DocumentAssembler() \
      .setInputCol("text") \
      .setOutputCol("document")
    
embeddings = E5Embeddings.pretrained("gpl_e5_base_test_1_aba799","en") \
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
    
val embeddings = E5Embeddings.pretrained("gpl_e5_base_test_1_aba799","en") 
    .setInputCols(Array("document")) 
    .setOutputCol("embeddings")

val pipeline = new Pipeline().setStages(Array(documentAssembler, embeddings))
val data = Seq("I love spark-nlp").toDF("text")
val pipelineModel = pipeline.fit(data)
val pipelineDF = pipelineModel.transform(data)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|gpl_e5_base_test_1_aba799|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document]|
|Output Labels:|[E5]|
|Language:|en|
|Size:|388.6 MB|

## References

https://huggingface.co/rithwik-db/gpl-e5-base-test-1-aba799