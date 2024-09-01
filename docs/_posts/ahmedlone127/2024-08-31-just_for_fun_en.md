---
layout: model
title: English just_for_fun CamemBertEmbeddings from Sci-fi-vy
author: John Snow Labs
name: just_for_fun
date: 2024-08-31
tags: [en, open_source, onnx, embeddings, camembert]
task: Embeddings
language: en
edition: Spark NLP 5.4.2
spark_version: 3.0
supported: true
engine: onnx
annotator: CamemBertEmbeddings
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained CamemBertEmbeddings model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`just_for_fun` is a English model originally trained by Sci-fi-vy.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/just_for_fun_en_5.4.2_3.0_1725136236649.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/just_for_fun_en_5.4.2_3.0_1725136236649.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

embeddings = CamemBertEmbeddings.pretrained("just_for_fun","en") \
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

val embeddings = CamemBertEmbeddings.pretrained("just_for_fun","en") 
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
|Model Name:|just_for_fun|
|Compatibility:|Spark NLP 5.4.2+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[camembert]|
|Language:|en|
|Size:|264.0 MB|

## References

https://huggingface.co/Sci-fi-vy/Just-For-Fun