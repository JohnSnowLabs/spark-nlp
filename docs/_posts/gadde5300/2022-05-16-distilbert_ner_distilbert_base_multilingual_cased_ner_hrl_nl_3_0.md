---
layout: model
title: Dutch Named Entity Recognition (from Davlan)
author: John Snow Labs
name: distilbert_ner_distilbert_base_multilingual_cased_ner_hrl
date: 2022-05-16
tags: [distilbert, ner, token_classification, nl, open_source]
task: Named Entity Recognition
language: nl
edition: Spark NLP 3.4.2
spark_version: 3.0
supported: true
annotator: DistilBertForTokenClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained Named Entity Recognition model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `distilbert-base-multilingual-cased-ner-hrl` is a Dutch model orginally trained by `Davlan`.

## Predicted Entities

`DATE`, `PER`, `LOC`, `ORG`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/distilbert_ner_distilbert_base_multilingual_cased_ner_hrl_nl_3.4.2_3.0_1652721841985.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/distilbert_ner_distilbert_base_multilingual_cased_ner_hrl_nl_3.4.2_3.0_1652721841985.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

tokenClassifier = DistilBertForTokenClassification.pretrained("distilbert_ner_distilbert_base_multilingual_cased_ner_hrl","nl") \
    .setInputCols(["sentence", "token"]) \
    .setOutputCol("ner")

pipeline = Pipeline(stages=[documentAssembler, sentenceDetector, tokenizer, tokenClassifier])

data = spark.createDataFrame([["Ik hou van Spark NLP"]]).toDF("text")

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

val tokenClassifier = DistilBertForTokenClassification.pretrained("distilbert_ner_distilbert_base_multilingual_cased_ner_hrl","nl") 
    .setInputCols(Array("sentence", "token")) 
    .setOutputCol("ner")

val pipeline = new Pipeline().setStages(Array(documentAssembler,sentenceDetector, tokenizer, tokenClassifier))

val data = Seq("Ik hou van Spark NLP").toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|distilbert_ner_distilbert_base_multilingual_cased_ner_hrl|
|Compatibility:|Spark NLP 3.4.2+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[ner]|
|Language:|nl|
|Size:|505.8 MB|
|Case sensitive:|true|
|Max sentence length:|128|

## References

- https://huggingface.co/Davlan/distilbert-base-multilingual-cased-ner-hrl
- https://camel.abudhabi.nyu.edu/anercorp/
- https://www.clips.uantwerpen.be/conll2003/ner/
- https://www.clips.uantwerpen.be/conll2003/ner/
- https://www.clips.uantwerpen.be/conll