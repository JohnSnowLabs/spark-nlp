---
layout: model
title: Igbo Named Entity Recognition (from Davlan)
author: John Snow Labs
name: bert_ner_bert_base_multilingual_cased_masakhaner
date: 2022-05-09
tags: [bert, ner, token_classification, ig, open_source]
task: Named Entity Recognition
language: ig
edition: Spark NLP 3.4.2
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained Named Entity Recognition model, uploaded to Hugging Face, adapted and imported into Spark NLP. `bert-base-multilingual-cased-masakhaner` is a Igbo model orginally trained by `Davlan`.

## Predicted Entities

`LOC`, `DATE`, `PER`, `ORG`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_ner_bert_base_multilingual_cased_masakhaner_ig_3.4.2_3.0_1652099895556.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = DocumentAssembler() \
        .setInputCol("text") \
        .setOutputCol("document")

sentenceDetector = SentenceDetectorDLModel.pretrained("sentence_detector_dl", "xx")\
       .setInputCols(["document"])\
       .setOutputCol("sentence")

tokenizer = Tokenizer() \
    .setInputCols("sentence") \
    .setOutputCol("token")

tokenClassifier = BertForTokenClassification.pretrained("bert_ner_bert_base_multilingual_cased_masakhaner","ig") \
    .setInputCols(["sentence", "token"]) \
    .setOutputCol("ner")

pipeline = Pipeline(stages=[documentAssembler, sentenceDetector, tokenizer, tokenClassifier])

data = spark.createDataFrame([["Ahụrụ m n'anya na-atọ m ụtọ"]]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val documentAssembler = new DocumentAssembler() 
          .setInputCol("text") 
          .setOutputCol("document")

val sentenceDetector = SentenceDetectorDLModel.pretrained("sentence_detector_dl", "xx")
       .setInputCols(Array("document"))
       .setOutputCol("sentence")

val tokenizer = new Tokenizer() 
    .setInputCols(Array("sentence"))
    .setOutputCol("token")

val tokenClassifier = BertForTokenClassification.pretrained("bert_ner_bert_base_multilingual_cased_masakhaner","ig") 
    .setInputCols(Array("sentence", "token")) 
    .setOutputCol("ner")

val pipeline = new Pipeline().setStages(Array(documentAssembler,sentenceDetector, tokenizer, tokenClassifier))

val data = Seq("Ahụrụ m n'anya na-atọ m ụtọ").toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_ner_bert_base_multilingual_cased_masakhaner|
|Compatibility:|Spark NLP 3.4.2+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[ner]|
|Language:|ig|
|Size:|665.6 MB|
|Case sensitive:|true|
|Max sentence length:|128|

## References

- https://huggingface.co/Davlan/bert-base-multilingual-cased-masakhaner
- https://github.com/masakhane-io/masakhane-ner
- https://github.com/masakhane-io/masakhane-ner
- https://arxiv.org/abs/2103.11811