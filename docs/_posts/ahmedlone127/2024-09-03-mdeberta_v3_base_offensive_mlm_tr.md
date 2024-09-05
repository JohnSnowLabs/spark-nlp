---
layout: model
title: Turkish mdeberta_v3_base_offensive_mlm DeBertaEmbeddings from Overfit-GM
author: John Snow Labs
name: mdeberta_v3_base_offensive_mlm
date: 2024-09-03
tags: [tr, open_source, onnx, embeddings, deberta]
task: Embeddings
language: tr
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

Pretrained DeBertaEmbeddings model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`mdeberta_v3_base_offensive_mlm` is a Turkish model originally trained by Overfit-GM.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/mdeberta_v3_base_offensive_mlm_tr_5.5.0_3.0_1725377292586.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/mdeberta_v3_base_offensive_mlm_tr_5.5.0_3.0_1725377292586.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

embeddings = DeBertaEmbeddings.pretrained("mdeberta_v3_base_offensive_mlm","tr") \
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

val embeddings = DeBertaEmbeddings.pretrained("mdeberta_v3_base_offensive_mlm","tr") 
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
|Model Name:|mdeberta_v3_base_offensive_mlm|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[deberta]|
|Language:|tr|
|Size:|1.0 GB|

## References

https://huggingface.co/Overfit-GM/mdeberta-v3-base-offensive-mlm