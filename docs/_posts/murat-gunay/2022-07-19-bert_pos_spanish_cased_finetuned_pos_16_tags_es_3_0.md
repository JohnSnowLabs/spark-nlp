---
layout: model
title: Spanish BertForTokenClassification Cased model (from mrm8488)
author: John Snow Labs
name: bert_pos_spanish_cased_finetuned_pos_16_tags
date: 2022-07-19
tags: [open_source, bert, ner, pos, es]
task: Part of Speech Tagging
language: es
edition: Spark NLP 4.0.0
spark_version: 3.0
supported: true
annotator: BertForTokenClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BERT POS model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `bert-spanish-cased-finetuned-pos-16-tags` is a Spanish model originally trained by `mrm8488`.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_pos_spanish_cased_finetuned_pos_16_tags_es_4.0.0_3.0_1658249425886.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")

sentenceDetector = SentenceDetector()\
    .setInputCols(["document"])\
    .setOutputCol("sentence")

tokenizer = Tokenizer() \
    .setInputCols("sentence") \
    .setOutputCol("token")
  
pos = BertForTokenClassification.pretrained("bert_pos_spanish_cased_finetuned_pos_16_tags","es") \
    .setInputCols(["sentence", "token"]) \
    .setOutputCol("pos")
    
pipeline = Pipeline(stages=[documentAssembler, tokenizer, pos])

data = spark.createDataFrame([["PUT YOUR STRING HERE."]]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val documentAssembler = new DocumentAssembler() 
      .setInputCol("text") 
      .setOutputCol("document")

val sentenceDetector = new SentenceDetector()
    .setInputCols(Array("document"))
    .setOutputCol("sentence")

val tokenizer = new Tokenizer() 
    .setInputCols(Array("sentence"))
    .setOutputCol("token")

val pos = BertForTokenClassification.pretrained("bert_pos_spanish_cased_finetuned_pos_16_tags","es") 
    .setInputCols(Array("sentence", "token")) 
    .setOutputCol("pos")

val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, pos))

val data = Seq("PUT YOUR STRING HERE.").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_pos_spanish_cased_finetuned_pos_16_tags|
|Compatibility:|Spark NLP 4.0.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[ner]|
|Language:|es|
|Size:|410.2 MB|
|Case sensitive:|true|
|Max sentence length:|128|

## References

https://huggingface.co/mrm8488/bert-spanish-cased-finetuned-pos-16-tags
