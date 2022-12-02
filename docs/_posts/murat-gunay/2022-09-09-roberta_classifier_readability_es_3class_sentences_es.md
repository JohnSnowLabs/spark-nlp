---
layout: model
title: Spanish RobertaForSequenceClassification Cased model (from hackathon-pln-es)
author: John Snow Labs
name: roberta_classifier_readability_es_3class_sentences
date: 2022-09-09
tags: [es, open_source, roberta, sequence_classification, classification]
task: Text Classification
language: es
edition: Spark NLP 4.1.0
spark_version: 3.0
supported: true
annotator: RoBertaForSequenceClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained RobertaForSequenceClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `readability-es-3class-sentences` is a Spanish model originally trained by `hackathon-pln-es`.

## Predicted Entities

`intermediate`, `basic`, `advanced`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/roberta_classifier_readability_es_3class_sentences_es_4.1.0_3.0_1662764976252.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = DocumentAssembler() \
    .setInputCols(["text"]) \
    .setOutputCols("document")

tokenizer = Tokenizer() \
    .setInputCols("document") \
    .setOutputCol("token")

seq_classifier = RoBertaForSequenceClassification.pretrained("roberta_classifier_readability_es_3class_sentences","es") \
    .setInputCols(["document", "token"]) \
    .setOutputCol("class")
    
pipeline = Pipeline(stages=[documentAssembler, tokenizer, seq_classifier])

data = spark.createDataFrame([["PUT YOUR STRING HERE"]]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val documentAssembler = new DocumentAssembler() 
      .setInputCols(Array("text")) 
      .setOutputCols(Array("document"))
      
val tokenizer = new Tokenizer()
    .setInputCols("document")
    .setOutputCol("token")
 
val seq_classifier = RoBertaForSequenceClassification.pretrained("roberta_classifier_readability_es_3class_sentences","es") 
    .setInputCols(Array("document", "token"))
    .setOutputCol("class")
   
val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, seq_classifier))

val data = Seq("PUT YOUR STRING HERE").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|roberta_classifier_readability_es_3class_sentences|
|Compatibility:|Spark NLP 4.1.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[class]|
|Language:|es|
|Size:|442.8 MB|
|Case sensitive:|true|
|Max sentence length:|256|

## References

- https://huggingface.co/hackathon-pln-es/readability-es-3class-sentences
- https://wandb.ai/readability-es/readability-es/reports/Texts-Readability-Analysis-for-Spanish--VmlldzoxNzU2MDUx
- https://wandb.ai/readability-es/readability-es/runs/1qe3kbqj/overview
- https://lmvasque.github.io/
- https://twitter.com/pcuenq
- https://www.fireblend.com/
- https://feralvam.github.io/