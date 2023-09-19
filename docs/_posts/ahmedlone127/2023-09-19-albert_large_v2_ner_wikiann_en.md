---
layout: model
title: English albert_large_v2_ner_wikiann AlbertForTokenClassification from Gladiator
author: John Snow Labs
name: albert_large_v2_ner_wikiann
date: 2023-09-19
tags: [albert, en, open_source, token_classification, onnx]
task: Named Entity Recognition
language: en
edition: Spark NLP 5.1.2
spark_version: 3.0
supported: true
engine: onnx
annotator: AlbertForTokenClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained AlbertForTokenClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`albert_large_v2_ner_wikiann` is a English model originally trained by Gladiator.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/albert_large_v2_ner_wikiann_en_5.1.2_3.0_1695087727017.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/albert_large_v2_ner_wikiann_en_5.1.2_3.0_1695087727017.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python


document_assembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("documents")
    
    
sequenceClassifier = AlbertForTokenClassification.pretrained("albert_large_v2_ner_wikiann","en") \
            .setInputCols(["documents","token"]) \
            .setOutputCol("ner")

pipeline = Pipeline().setStages([document_assembler, sequenceClassifier])

pipelineModel = pipeline.fit(data)

pipelineDF = pipelineModel.transform(data)

```
```scala


val document_assembler = new DocumentAssembler()
    .setInputCol("text") 
    .setOutputCol("embeddings")
    
val sequenceClassifier = AlbertForTokenClassification  
    .pretrained("albert_large_v2_ner_wikiann", "en")
    .setInputCols(Array("documents","token")) 
    .setOutputCol("ner") 

val pipeline = new Pipeline().setStages(Array(document_assembler, sequenceClassifier))

val pipelineModel = pipeline.fit(data)

val pipelineDF = pipelineModel.transform(data)


```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|albert_large_v2_ner_wikiann|
|Compatibility:|Spark NLP 5.1.2+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[documents, token]|
|Output Labels:|[ner]|
|Language:|en|
|Size:|62.8 MB|

## References

https://huggingface.co/Gladiator/albert-large-v2_ner_wikiann