---
layout: model
title: English sentence_transformers_all_mpnet_base_v2_2epoch_100pair_mar2_contrastiveloss_morality MPNetEmbeddings from ahessamb
author: John Snow Labs
name: sentence_transformers_all_mpnet_base_v2_2epoch_100pair_mar2_contrastiveloss_morality
date: 2024-09-11
tags: [en, open_source, onnx, embeddings, mpnet]
task: Embeddings
language: en
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
engine: onnx
annotator: MPNetEmbeddings
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained MPNetEmbeddings model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`sentence_transformers_all_mpnet_base_v2_2epoch_100pair_mar2_contrastiveloss_morality` is a English model originally trained by ahessamb.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/sentence_transformers_all_mpnet_base_v2_2epoch_100pair_mar2_contrastiveloss_morality_en_5.5.0_3.0_1726034219892.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/sentence_transformers_all_mpnet_base_v2_2epoch_100pair_mar2_contrastiveloss_morality_en_5.5.0_3.0_1726034219892.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
 
documentAssembler = DocumentAssembler() \
      .setInputCol("text") \
      .setOutputCol("document")
    
embeddings = MPNetEmbeddings.pretrained("sentence_transformers_all_mpnet_base_v2_2epoch_100pair_mar2_contrastiveloss_morality","en") \
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
    
val embeddings = MPNetEmbeddings.pretrained("sentence_transformers_all_mpnet_base_v2_2epoch_100pair_mar2_contrastiveloss_morality","en") 
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
|Model Name:|sentence_transformers_all_mpnet_base_v2_2epoch_100pair_mar2_contrastiveloss_morality|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document]|
|Output Labels:|[mpnet]|
|Language:|en|
|Size:|406.8 MB|

## References

https://huggingface.co/ahessamb/sentence-transformers-all-mpnet-base-v2-2epoch-100pair-mar2-contrastiveloss-morality