---
layout: model
title: English CamembertForTokenClassification Cased model (from HueyNemud)
author: John Snow Labs
name: camembert_classifier_das22_43_pretrained_finetuned_pero
date: 2022-09-25
tags: [camembert, ner, open_source, en]
task: Named Entity Recognition
language: en
nav_key: models
edition: Spark NLP 4.2.0
spark_version: 3.0
supported: true
annotator: CamemBertForTokenClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained CamembertForTokenClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `das22-43-camembert_pretrained_finetuned_pero` is a English model originally trained by `HueyNemud`.

## Predicted Entities

`TITRE`, `MISC`, `ACT`, `FT`, `LOC`, `ORG`, `PER`, `CARDINAL`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/camembert_classifier_das22_43_pretrained_finetuned_pero_en_4.2.0_3.0_1664084297524.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/camembert_classifier_das22_43_pretrained_finetuned_pero_en_4.2.0_3.0_1664084297524.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

sequenceClassifier_loaded = CamemBertForTokenClassification.pretrained("camembert_classifier_das22_43_pretrained_finetuned_pero","en") \
    .setInputCols(["sentence", "token"]) \
    .setOutputCol("ner")

pipeline = Pipeline(stages=[documentAssembler,sentenceDetector,tokenizer,sequenceClassifier_loaded])

data = spark.createDataFrame([["PUT YOUR STRING HERE"]]).toDF("text")

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

val sequenceClassifier_loaded = CamemBertForTokenClassification.pretrained("camembert_classifier_das22_43_pretrained_finetuned_pero","en") 
    .setInputCols(Array("sentence", "token")) 
    .setOutputCol("ner")

val pipeline = new Pipeline().setStages(Array(documentAssembler,sentenceDetector,tokenizer,sequenceClassifier_loaded))

val data = Seq("PUT YOUR STRING HERE").toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|camembert_classifier_das22_43_pretrained_finetuned_pero|
|Compatibility:|Spark NLP 4.2.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[class]|
|Language:|en|
|Size:|413.4 MB|
|Case sensitive:|true|
|Max sentence length:|256|

## References

- https://huggingface.co/HueyNemud/das22-43-camembert_pretrained_finetuned_pero
- https://doi.org/10.1007/978-3-031-06555-2_30
- https://github.com/soduco/paper-ner-bench-das22