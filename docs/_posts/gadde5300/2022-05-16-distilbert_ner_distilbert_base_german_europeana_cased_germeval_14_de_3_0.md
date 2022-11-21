---
layout: model
title: German Named Entity Recognition (from HuggingAlex1247)
author: John Snow Labs
name: distilbert_ner_distilbert_base_german_europeana_cased_germeval_14
date: 2022-05-16
tags: [distilbert, ner, token_classification, de, open_source]
task: Named Entity Recognition
language: de
edition: Spark NLP 3.4.2
spark_version: 3.0
supported: true
annotator: DistilBertForTokenClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained Named Entity Recognition model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `distilbert-base-german-europeana-cased-germeval_14` is a German model orginally trained by `HuggingAlex1247`.

## Predicted Entities

`LOCderiv`, `PERderiv`, `OTHderiv`, `OTH`, `OTHpart`, `ORGpart`, `PER`, `LOC`, `ORG`, `PERpart`, `ORGderiv`, `LOCpart`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/distilbert_ner_distilbert_base_german_europeana_cased_germeval_14_de_3.4.2_3.0_1652721706374.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

tokenClassifier = DistilBertForTokenClassification.pretrained("distilbert_ner_distilbert_base_german_europeana_cased_germeval_14","de") \
    .setInputCols(["sentence", "token"]) \
    .setOutputCol("ner")

pipeline = Pipeline(stages=[documentAssembler, sentenceDetector, tokenizer, tokenClassifier])

data = spark.createDataFrame([["Ich liebe Spark NLP"]]).toDF("text")

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

val tokenClassifier = DistilBertForTokenClassification.pretrained("distilbert_ner_distilbert_base_german_europeana_cased_germeval_14","de") 
    .setInputCols(Array("sentence", "token")) 
    .setOutputCol("ner")

val pipeline = new Pipeline().setStages(Array(documentAssembler,sentenceDetector, tokenizer, tokenClassifier))

val data = Seq("Ich liebe Spark NLP").toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|distilbert_ner_distilbert_base_german_europeana_cased_germeval_14|
|Compatibility:|Spark NLP 3.4.2+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[ner]|
|Language:|de|
|Size:|253.5 MB|
|Case sensitive:|true|
|Max sentence length:|128|

## References

- https://huggingface.co/HuggingAlex1247/distilbert-base-german-europeana-cased-germeval_14
- https://paperswithcode.com/sota?task=Token+Classification&dataset=germeval_14