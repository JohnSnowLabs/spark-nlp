---
layout: model
title: Classify text about Effective, Renewal or Termination date
author: John Snow Labs
name: legclf_dates_sm
date: 2022-11-21
tags: [effective, renewal, termination, date, en, licensed]
task: Text Classification
language: en
edition: Legal NLP 1.0.0
spark_version: 3.0
supported: true
annotator: LegalClassifierDLModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This is a Text Classification model can help you classify if a paragraph talks about an Effective Date, a Renewal Date, a Termination Date or something else. Don't confuse this model with the NER model (`legner_dates_sm`) which allows you to extract the actual dates from the texts.

## Predicted Entities

`EFFECTIVE_DATE`, `RENEWAL_DATE`, `TERMINATION_DATE`, `other`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/legal/models/legclf_dates_sm_en_1.0.0_3.0_1669034322560.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
documentAssembler = DocumentAssembler() \
  .setInputCol("text") \
  .setOutputCol("document")

embeddings = BertSentenceEmbeddings.pretrained("sent_bert_base_cased", "en") \
  .setInputCols("document") \
  .setOutputCol("sentence_embeddings")

docClassifier = legal.ClassifierDLModel.pretrained('legclf_dates_sm', 'en', 'legal/models')\
    .setInputCols(["sentence_embeddings"])\
    .setOutputCol("label")

nlpPipeline = Pipeline(stages=[
    documentAssembler, 
    embeddings,
    docClassifier])

text = ["""Renewal Date means January 1, 2018."""]

empty_data = spark.createDataFrame([[""]]).toDF("text")

model = nlpPipeline.fit(empty_data)

res = model.transform(spark.createDataFrame([text]).toDF("text"))
```

</div>

## Results

```bash
+--------------+
|        result|
+--------------+
|[RENEWAL_DATE]|
+--------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|legclf_dates_sm|
|Compatibility:|Legal NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence_embeddings]|
|Output Labels:|[label]|
|Language:|en|
|Size:|22.5 MB|

## References

In-house annotations.

## Benchmarking

```bash
           label  precision    recall  f1-score   support
  EFFECTIVE_DATE       1.00      0.80      0.89         5
    RENEWAL_DATE       1.00      1.00      1.00         6
TERMINATION_DATE       0.86      0.75      0.80         8
           other       0.91      1.00      0.95        21
        accuracy          -         -      0.93        40
       macro-avg       0.94      0.89      0.91        40
    weighted-avg       0.93      0.93      0.92        40
```