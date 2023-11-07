---
layout: model
title: Tagalog bert_tagalog_base_uncased_sayula_popoluca_tagger BertForTokenClassification from syke9p3
author: John Snow Labs
name: bert_tagalog_base_uncased_sayula_popoluca_tagger
date: 2023-11-07
tags: [bert, tl, open_source, token_classification, onnx]
task: Named Entity Recognition
language: tl
edition: Spark NLP 5.2.0
spark_version: 3.0
supported: true
engine: onnx
annotator: BertForTokenClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForTokenClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`bert_tagalog_base_uncased_sayula_popoluca_tagger` is a Tagalog model originally trained by syke9p3.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_tagalog_base_uncased_sayula_popoluca_tagger_tl_5.2.0_3.0_1699388445151.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_tagalog_base_uncased_sayula_popoluca_tagger_tl_5.2.0_3.0_1699388445151.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python


documentAssembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("documents")
    
    
tokenClassifier = BertForTokenClassification.pretrained("bert_tagalog_base_uncased_sayula_popoluca_tagger","tl") \
            .setInputCols(["documents","token"]) \
            .setOutputCol("ner")

pipeline = Pipeline().setStages([documentAssembler, tokenClassifier])

pipelineModel = pipeline.fit(data)

pipelineDF = pipelineModel.transform(data)

```
```scala


val documentAssembler = new DocumentAssembler()
    .setInputCol("text") 
    .setOutputCol("embeddings")
    
val tokenClassifier = BertForTokenClassification  
    .pretrained("bert_tagalog_base_uncased_sayula_popoluca_tagger", "tl")
    .setInputCols(Array("documents","token")) 
    .setOutputCol("ner") 

val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenClassifier))

val pipelineModel = pipeline.fit(data)

val pipelineDF = pipelineModel.transform(data)


```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_tagalog_base_uncased_sayula_popoluca_tagger|
|Compatibility:|Spark NLP 5.2.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[documents, token]|
|Output Labels:|[ner]|
|Language:|tl|
|Size:|470.3 MB|

## References

https://huggingface.co/syke9p3/bert-tagalog-base-uncased-pos-tagger