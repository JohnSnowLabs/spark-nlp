---
layout: model
title: French camembert_bio_base CamemBertEmbeddings from almanach
author: John Snow Labs
name: camembert_bio_base
date: 2025-06-22
tags: [fr, open_source, onnx, embeddings, camembert, openvino]
task: Embeddings
language: fr
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
engine: openvino
annotator: CamemBertEmbeddings
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained CamemBertEmbeddings model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`camembert_bio_base` is a French model originally trained by almanach.

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/camembert_bio_base_fr_5.5.1_3.0_1750618262643.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/camembert_bio_base_fr_5.5.1_3.0_1750618262643.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

embeddings = CamemBertEmbeddings.pretrained("camembert_bio_base","fr") \
      .setInputCols(["document", "token"]) \
      .setOutputCol("embeddings")       
        
pipeline = Pipeline().setStages([documentAssembler, tokenizer, embeddings])
data = spark.createDataFrame([["I love spark-nlp"]]).toDF("text")
pipelineModel = pipeline.fit(data)
pipelineDF = pipelineModel.transform(data)
```
```scala
val documentAssembler = new DocumentAssembler() 
    .setInputCol("text") 
    .setOutputCol("document")
    
val tokenizer = new Tokenizer() 
    .setInputCols(Array("document"))
    .setOutputCol("token")

val embeddings = CamemBertEmbeddings.pretrained("camembert_bio_base","fr") 
    .setInputCols(Array("document", "token")) 
    .setOutputCol("embeddings")

val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, embeddings))
val data = Seq("I love spark-nlp").toDF("text")
val pipelineModel = pipeline.fit(data)
val pipelineDF = pipelineModel.transform(data)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|camembert_bio_base|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[camembert]|
|Language:|fr|
|Size:|412.8 MB|

## References

References

https://huggingface.co/almanach/camembert-bio-base