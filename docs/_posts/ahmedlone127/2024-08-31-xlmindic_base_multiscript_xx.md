---
layout: model
title: Multilingual xlmindic_base_multiscript AlbertEmbeddings from ibraheemmoosa
author: John Snow Labs
name: xlmindic_base_multiscript
date: 2024-08-31
tags: [xx, open_source, onnx, embeddings, albert]
task: Embeddings
language: xx
edition: Spark NLP 5.4.2
spark_version: 3.0
supported: true
engine: onnx
annotator: AlbertEmbeddings
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained AlbertEmbeddings model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`xlmindic_base_multiscript` is a Multilingual model originally trained by ibraheemmoosa.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/xlmindic_base_multiscript_xx_5.4.2_3.0_1725134901955.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/xlmindic_base_multiscript_xx_5.4.2_3.0_1725134901955.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

embeddings = AlbertEmbeddings.pretrained("xlmindic_base_multiscript","xx") \
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

val embeddings = AlbertEmbeddings.pretrained("xlmindic_base_multiscript","xx") 
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
|Model Name:|xlmindic_base_multiscript|
|Compatibility:|Spark NLP 5.4.2+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[albert]|
|Language:|xx|
|Size:|51.7 MB|

## References

https://huggingface.co/ibraheemmoosa/xlmindic-base-multiscript