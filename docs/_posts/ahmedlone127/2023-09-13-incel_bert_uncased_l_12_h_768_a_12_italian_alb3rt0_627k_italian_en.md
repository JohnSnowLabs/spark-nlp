---
layout: model
title: English incel_bert_uncased_l_12_h_768_a_12_italian_alb3rt0_627k_italian BertEmbeddings from pgajo
author: John Snow Labs
name: incel_bert_uncased_l_12_h_768_a_12_italian_alb3rt0_627k_italian
date: 2023-09-13
tags: [bert, en, open_source, fill_mask, onnx]
task: Embeddings
language: en
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

Pretrained BertEmbeddings  model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`incel_bert_uncased_l_12_h_768_a_12_italian_alb3rt0_627k_italian` is a English model originally trained by pgajo.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/incel_bert_uncased_l_12_h_768_a_12_italian_alb3rt0_627k_italian_en_5.1.1_3.0_1694621049338.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/incel_bert_uncased_l_12_h_768_a_12_italian_alb3rt0_627k_italian_en_5.1.1_3.0_1694621049338.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python


document_assembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("documents")
    
    
embeddings =BertEmbeddings.pretrained("incel_bert_uncased_l_12_h_768_a_12_italian_alb3rt0_627k_italian","en") \
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
    .pretrained("incel_bert_uncased_l_12_h_768_a_12_italian_alb3rt0_627k_italian", "en")
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
|Model Name:|incel_bert_uncased_l_12_h_768_a_12_italian_alb3rt0_627k_italian|
|Compatibility:|Spark NLP 5.1.1+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[documents, token]|
|Output Labels:|[embeddings]|
|Language:|en|
|Size:|688.7 MB|

## References

https://huggingface.co/pgajo/incel-bert_uncased_L-12_H-768_A-12_italian_alb3rt0-627k_italian