---
layout: model
title: English RobertaForSequenceClassification Cased model (from MoritzLaurer)
author: John Snow Labs
name: roberta_classifier_covid_policy_21
date: 2022-09-09
tags: [en, open_source, roberta, sequence_classification, classification]
task: Text Classification
language: en
edition: Spark NLP 4.1.0
spark_version: 3.0
supported: true
annotator: RoBertaForSequenceClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained RobertaForSequenceClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `covid-policy-roberta-21` is a English model originally trained by `MoritzLaurer`.

## Predicted Entities

`Quarantine`, `Health Monitoring`, `Lockdown`, `Restrictions of Mass Gatherings`, `Health Testing`, `Public Awareness Measures`, `Closure and Regulation of Schools`, `New Task Force, Bureau or Administrative Configuration`, `Restriction and Regulation of Businesses`, `COVID-19 Vaccines`, `Other Policy Not Listed Above`, `Internal Border Restrictions`, `Restriction and Regulation of Government Services`, `Curfew`, `Social Distancing`, `Health Resources`, `External Border Restrictions`, `Anti-Disinformation Measures`, `Hygiene`, `Other`, `Declaration of Emergency`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/roberta_classifier_covid_policy_21_en_4.1.0_3.0_1662763263282.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/roberta_classifier_covid_policy_21_en_4.1.0_3.0_1662763263282.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

seq_classifier = RoBertaForSequenceClassification.pretrained("roberta_classifier_covid_policy_21","en") \
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
 
val seq_classifier = RoBertaForSequenceClassification.pretrained("roberta_classifier_covid_policy_21","en") 
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
|Model Name:|roberta_classifier_covid_policy_21|
|Compatibility:|Spark NLP 4.1.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[class]|
|Language:|en|
|Size:|309.4 MB|
|Case sensitive:|true|
|Max sentence length:|256|

## References

- https://huggingface.co/MoritzLaurer/covid-policy-roberta-21
- https://www.ceps.eu/ceps-staff/moritz-laurer/