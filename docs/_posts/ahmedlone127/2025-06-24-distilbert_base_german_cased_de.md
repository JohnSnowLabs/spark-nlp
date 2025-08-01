---
layout: model
title: German distilbert_base_german_cased DistilBertEmbeddings from huggingface
author: John Snow Labs
name: distilbert_base_german_cased
date: 2025-06-24
tags: [distilbert, de, open_source, fill_mask, onnx, openvino]
task: Embeddings
language: de
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
engine: openvino
annotator: DistilBertEmbeddings
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained DistilBertEmbeddings  model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`distilbert_base_german_cased` is a German model originally trained by huggingface.

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/distilbert_base_german_cased_de_5.5.1_3.0_1750780410389.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/distilbert_base_german_cased_de_5.5.1_3.0_1750780410389.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
document_assembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("documents")
    
    
embeddings =DistilBertEmbeddings.pretrained("distilbert_base_german_cased","de") \
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
    
val embeddings = DistilBertEmbeddings 
    .pretrained("distilbert_base_german_cased", "de")
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
|Model Name:|distilbert_base_german_cased|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[distilbert]|
|Language:|de|
|Size:|250.3 MB|

## References

References

References

https://huggingface.co/distilbert-base-german-cased