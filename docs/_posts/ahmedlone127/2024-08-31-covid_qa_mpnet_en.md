---
layout: model
title: English covid_qa_mpnet MPNetEmbeddings from shaina
author: John Snow Labs
name: covid_qa_mpnet
date: 2024-08-31
tags: [mpnet, en, open_source, onnx]
task: Embeddings
language: en
edition: Spark NLP 5.4.2
spark_version: 3.0
supported: true
engine: onnx
annotator: MPNetForQuestionAnswering
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained MPNetEmbeddings  model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`covid_qa_mpnet` is a English model originally trained by shaina.

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/covid_qa_mpnet_en_5.4.2_3.0_1725115523061.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/covid_qa_mpnet_en_5.4.2_3.0_1725115523061.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
document_assembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("documents")
    
    
embeddings =MPNetEmbeddings.pretrained("covid_qa_mpnet","en") \
            .setInputCols(["documents"]) \
            .setOutputCol("mpnet_embeddings")

pipeline = Pipeline().setStages([document_assembler, embeddings])

pipelineModel = pipeline.fit(data)

pipelineDF = pipelineModel.transform(data)
```
```scala
val document_assembler = new DocumentAssembler()
    .setInputCol("text") 
    .setOutputCol("documents")
    
val embeddings = MPNetEmbeddings 
    .pretrained("covid_qa_mpnet", "en")
    .setInputCols(Array("documents")) 
    .setOutputCol("mpnet_embeddings") 

val pipeline = new Pipeline().setStages(Array(document_assembler, embeddings))

val pipelineModel = pipeline.fit(data)

val pipelineDF = pipelineModel.transform(data)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|covid_qa_mpnet|
|Compatibility:|Spark NLP 5.4.2+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document_question, document_context]|
|Output Labels:|[answer]|
|Language:|en|
|Size:|399.6 MB|

## References

References

https://huggingface.co/shaina/covid_qa_mpnet