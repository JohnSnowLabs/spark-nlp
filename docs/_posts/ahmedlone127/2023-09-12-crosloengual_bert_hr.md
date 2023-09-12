---
layout: model
title: Croatian crosloengual_bert BertEmbeddings from EMBEDDIA
author: John Snow Labs
name: crosloengual_bert
date: 2023-09-12
tags: [bert, hr, open_source, fill_mask, onnx]
task: Embeddings
language: hr
edition: Spark NLP 5.1.1
spark_version: 3.0
supported: true
engine: onnx
annotator: BertEmbeddings
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertEmbeddings  model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`crosloengual_bert` is a Croatian model originally trained by EMBEDDIA.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/crosloengual_bert_hr_5.1.1_3.0_1694509284745.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/crosloengual_bert_hr_5.1.1_3.0_1694509284745.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python


document_assembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("documents")
    
    
embeddings =BertEmbeddings.pretrained("crosloengual_bert","hr") \
            .setInputCols(["documents","token"]) \
            .setOutputCol("embeddings")

pipeline = Pipeline().setStages([document_assembler, embeddings])

pipelineModel = pipeline.fit(data)

pipelineDF = pipelineModel.transform(data)

```
```scala


val document_assembler = new DocumentAssembler()
    .setInputCol("text") 
    .setOutputCol("embeddings")
    
val embeddings = BertEmbeddings 
    .pretrained("crosloengual_bert", "hr")
    .setInputCols(Array("documents","token")) 
    .setOutputCol("embeddings") 

val pipeline = new Pipeline().setStages(Array(document_assembler, embeddings))

val pipelineModel = pipeline.fit(data)

val pipelineDF = pipelineModel.transform(data)


```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|crosloengual_bert|
|Compatibility:|Spark NLP 5.1.1+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[documents, token]|
|Output Labels:|[embeddings]|
|Language:|hr|
|Size:|463.4 MB|

## References

https://huggingface.co/EMBEDDIA/crosloengual-bert