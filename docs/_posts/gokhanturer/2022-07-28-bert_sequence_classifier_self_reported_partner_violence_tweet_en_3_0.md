---
layout: model
title: Classification of Self-Reported Intimate Partner Violence (BioBERT)
author: John Snow Labs
name: bert_sequence_classifier_self_reported_partner_violence_tweet
date: 2022-07-28
tags: [sequence_classification, bert, classifier, clinical, en, licensed, public_health, partner_violence, tweet]
task: Text Classification
language: en
edition: Healthcare NLP 4.0.0
spark_version: 3.0
supported: true
annotator: MedicalBertForSequenceClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Classification of Self-Reported Intimate Partner Violence on Twitter. This model involves the detection the potential IPV victims on social media platforms (in English tweets).

## Predicted Entities

`intimate_partner_violence`, `non-intimate_partner_violence`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/PUBLIC_HEALTH_PARTNER_VIOLENCE/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/PUBLIC_HEALTH_MB4SC.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/bert_sequence_classifier_self_reported_partner_violence_tweet_en_4.0.0_3.0_1658999356448.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
document_assembler = DocumentAssembler() \
    .setInputCol('text') \
    .setOutputCol('document')

tokenizer = Tokenizer() \
    .setInputCols(['document']) \
    .setOutputCol('token')

sequenceClassifier = MedicalBertForSequenceClassification.pretrained("bert_sequence_classifier_self_reported_partner_violence_tweet", "en", "clinical/models")\
    .setInputCols(["document",'token'])\
    .setOutputCol("class")

pipeline = Pipeline(stages=[
    document_assembler, 
    tokenizer,
    sequenceClassifier    
])

example = spark.createDataFrame(["I am fed up with this toxic relation.I hate my husband.",
                              "Can i say something real quick I ve never been one to publicly drag an ex partner and sometimes I regret that. I ve been reflecting on the harm, abuse and violence that was done to me and those bitches are truly lucky I chose peace amp therapy because they are trash forreal."], StringType()).toDF("text")

result = pipeline.fit(example).transform(example)

result.select("text", "class.result").show(truncate=False)
```
```scala
val document_assembler = new DocumentAssembler() 
    .setInputCol("text") 
    .setOutputCol("document")

val tokenizer = new Tokenizer() 
    .setInputCols(Array("document")) 
    .setOutputCol("token")

val sequenceClassifier = MedicalBertForSequenceClassification.pretrained("bert_sequence_classifier_self_reported_partner_violence_tweet", "en", "clinical/models")
  .setInputCols(Array("document","token"))
  .setOutputCol("class")

val pipeline = new Pipeline().setStages(Array(document_assembler, tokenizer, sequenceClassifier))

# couple of simple examples
val example = Seq(Array("I am fed up with this toxic relation.I hate my husband.",
                        "Can i say something real quick I ve never been one to publicly drag an ex partner and sometimes I regret that. I ve been reflecting on the harm, abuse and violence that was done to me and those bitches are truly lucky I chose peace amp therapy because they are trash forreal.")).toDF("text")

val result = pipeline.fit(example).transform(example)
```
</div>

## Results

```bash
+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+-------------------------------+
|text                                                                                                                                                                                                                                                                               |result                         |
+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+-------------------------------+
|I am fed up with this toxic relation.I hate my husband.                                                                                                                                                                                                                            |[non-intimate_partner_violence]|
|Can i say something real quick I ve never been one to publicly drag an ex partner and sometimes I regret that. I ve been reflecting on the harm, abuse and violence that was done to me and those bitches are truly lucky I chose peace amp therapy because they are trash forreal.|[intimate_partner_violence]    |
+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+-------------------------------+

```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_sequence_classifier_self_reported_partner_violence_tweet|
|Compatibility:|Healthcare NLP 4.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[class]|
|Language:|en|
|Size:|406.5 MB|
|Case sensitive:|true|
|Max sentence length:|128|

## References

[SMM4H 2022](https://healthlanguageprocessing.org/smm4h-2022/)

## Benchmarking

```bash
                        label  precision    recall  f1-score   support
    intimate_partner_violence       0.96      0.97      0.97       630
non-intimate_partner_violence       0.75      0.69      0.72        78
                     accuracy       -         -         0.94       708
                    macro-avg       0.86      0.83      0.84       708
                 weighted-avg       0.94      0.94      0.94       708
```
