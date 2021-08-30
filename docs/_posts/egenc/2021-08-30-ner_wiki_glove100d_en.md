---
layout: model
title: Detect Persons, Locations, Organizations and Misc Entities - EN (Wiki NER 6B 100)
author: John Snow Labs
name: ner_wiki_glove100d
date: 2021-08-30
tags: [open_source, ner, en]
task: Named Entity Recognition
language: en
edition: Spark NLP 3.2.0
spark_version: 2.4
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Wiki NER is a Named Entity Recognition (or NER) model, that can be used to find features such as names of people, places, and organizations. This NER model does not read words directly but instead reads word embeddings, which represent words as points such that more semantically similar words are closer together. Wiki NER 6B 100 is trained with GloVe 6B 100 word embeddings, so be sure to use the same embeddings in the pipeline.

## Predicted Entities

Persons, Locations, Organizations, Misc

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/public/NER_DE/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/NER_DE.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/ner_wiki_glove100d_en_3.2.0_2.4_1630317807304.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use

nerTagger = NerDLApproach()\
      .setInputCols(["sentence", "token", "embeddings"])\
      .setLabelColumn("label")\
      .setOutputCol("ner")\
      .setMaxEpochs(1)\
      .setLr(0.003)\
      .setBatchSize(32)\
      .setRandomSeed(0)\
      .setVerbose(1)\
      .setValidationSplit(0.2)\
      .setEvaluationLogExtended(True) \
      .setEnableOutputLogs(True)\
      .setIncludeConfidence(True)\
      .setOutputLogsPath('ner_logs') # if not set, logs will be written to ~/annotator_logs
 #    .setGraphFolder('graphs') >> put your graph file (pb) under this folder if you are using a custom graph generated thru 4.1 NerDL-Graph.ipynb notebook
 #    .setEnableMemoryOptimizer() >> if you have a limited memory and a large conll file, you can set this True to train batch by batch 
    
ner_pipeline = Pipeline(stages=[
      glove_embeddings,
      nerTagger
 ])

ner_model = ner_pipeline.fit(training_data)

predictions = ner_model.transform(test_data)

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
nerTagger = NerDLApproach()\
      .setInputCols(["sentence", "token", "embeddings"])\
      .setLabelColumn("label")\
      .setOutputCol("ner")\
      .setMaxEpochs(1)\
      .setLr(0.003)\
      .setBatchSize(32)\
      .setRandomSeed(0)\
      .setVerbose(1)\
      .setValidationSplit(0.2)\
      .setEvaluationLogExtended(True) \
      .setEnableOutputLogs(True)\
      .setIncludeConfidence(True)\
      .setOutputLogsPath('ner_logs') # if not set, logs will be written to ~/annotator_logs
 #    .setGraphFolder('graphs') >> put your graph file (pb) under this folder if you are using a custom graph generated thru 4.1 NerDL-Graph.ipynb notebook
 #    .setEnableMemoryOptimizer() >> if you have a limited memory and a large conll file, you can set this True to train batch by batch 
    
ner_pipeline = Pipeline(stages=[
      glove_embeddings,
      nerTagger
 ])

ner_model = ner_pipeline.fit(training_data)

predictions = ner_model.transform(test_data)
```
```scala
val ner = NerDLModel.pretrained("wikiner_6B_100", "de")
        .setInputCols(Array("document", "token", "embeddings"))
        .setOutputCol("ner")
```
</div>

## Results

```bash
+--------------+------------+----------+
|token         |ground_truth|prediction|
+--------------+------------+----------+
|CRICKET       |O           |O         |
|-             |O           |O         |
|LEICESTERSHIRE|B-ORG       |B-ORG     |
|TAKE          |O           |O         |
|OVER          |O           |O         |
|AT            |O           |O         |
|TOP           |O           |O         |
|AFTER         |O           |O         |
|INNINGS       |O           |O         |
|VICTORY       |O           |O         |
|.             |O           |O         |
|LONDON        |B-LOC       |B-LOC     |
|1996-08-30    |O           |O         |
|West          |B-MISC      |B-MISC    |
|Indian        |I-MISC      |I-MISC    |
|all-rounder   |O           |O         |
|Phil          |B-PER       |B-PER     |
|Simmons       |I-PER       |I-PER     |
|took          |O           |O         |
|four          |O           |O         |
+--------------+------------+----------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_wiki_glove100d|
|Type:|ner|
|Compatibility:|Spark NLP 3.2.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|en|
|Dependencies:|Glove_100d|

## Data Source

The model is trained based on data from https://en.wikipedia.org

## Benchmarking

```bash
  precision    recall  f1-score   support

       B-LOC       0.90      0.94      0.92      1837
      B-MISC       0.88      0.84      0.86       922
       B-ORG       0.91      0.80      0.85      1341
       B-PER       0.92      0.98      0.95      1842
       I-LOC       0.79      0.75      0.77       257
      I-MISC       0.84      0.56      0.68       346
       I-ORG       0.85      0.68      0.76       751
       I-PER       0.94      0.97      0.96      1307
           O       0.99      1.00      0.99     42759

    accuracy                           0.98     51362
   macro avg       0.89      0.84      0.86     51362
weighted avg       0.98      0.98      0.98     51362

```