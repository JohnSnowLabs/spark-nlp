---
layout: model
title: Multilingual DistilBertForTokenClassification Base Cased model (from mrm8488)
author: John Snow Labs
name: distilbert_ner_base_multi_cased_finetuned_typo_detection
date: 2022-07-21
tags: [open_source, distilbert, ner, typo, multilingual, xx]
task: Named Entity Recognition
language: xx
edition: Spark NLP 4.0.0
spark_version: 3.0
supported: true
annotator: DistilBertForTokenClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained DistilBERT NER model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `distilbert-base-multi-cased-finetuned-typo-detection` is a Multilingual model originally trained by `mrm8488`.

## Predicted Entities

`ok`, `typo`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/distilbert_ner_base_multi_cased_finetuned_typo_detection_xx_4.0.0_3.0_1658399913400.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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
  
ner = DistilBertForTokenClassification.pretrained("distilbert_ner_base_multi_cased_finetuned_typo_detection","xx") \
    .setInputCols(["sentence", "token"]) \
    .setOutputCol("ner")
    
pipeline = Pipeline(stages=[documentAssembler, tokenizer, ner])

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

val ner = DistilBertForTokenClassification.pretrained("distilbert_ner_base_multi_cased_finetuned_typo_detection","xx") 
    .setInputCols(Array("sentence", "token")) 
    .setOutputCol("ner")

val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, ner))

val data = Seq("PUT YOUR STRING HERE.").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|distilbert_ner_base_multi_cased_finetuned_typo_detection|
|Compatibility:|Spark NLP 4.0.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[ner]|
|Language:|xx|
|Size:|505.8 MB|
|Case sensitive:|true|
|Max sentence length:|128|

## References

https://huggingface.co/mrm8488/distilbert-base-multi-cased-finetuned-typo-detection