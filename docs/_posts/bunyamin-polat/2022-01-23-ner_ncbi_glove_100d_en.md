---
layout: model
title: Detect DISEASE Entities in English Clinical Data
author: bunyamin-polat
name: ner_ncbi_glove_100d
date: 2022-01-23
tags: [ner, open_source, ncbi, en]
task: Named Entity Recognition
language: en
edition: Spark NLP 3.4.0
spark_version: 3.0
supported: false
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

NCBI-NER is Named Entity Recognition (or NER) model, meaning it annotates text to find the name of disease. This NER model does not read words diractly but instead reads word embeddings, which represent words as vectors such that more semanticially words are closer together. NCBI-NER is trained with Glove_100d word emnbeddings, so be sure to use the same embeddings in the pipeline

## Predicted Entities

`DISEASE`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://colab.research.google.com/github/bunyamin-polat/Spark-NLP-NER-Model-with-NCBI-disease/blob/main/NER_Model_Training.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/community.johnsnowlabs.com/bunyamin-polat/ner_ncbi_glove_100d_en_3.4.0_3.0_1642980603588.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
glove_embeddings = WordEmbeddingsModel.pretrained()\
    .setInputCols(["sentence", "token"])\
    .setOutputCol("embeddings")

nerTagger = NerDLApproach()\
    .setInputCols(["sentence", "token", "embeddings"])\
    .setLabelColumn("label")\
    .setOutputCol("ner")\
    .setMaxEpochs(14)\
    .setLr(0.003)\
    .setDropout(0.5)\
    .setBatchSize(10)\
    .setRandomSeed(0)\
    .setValidationSplit(0.2)\
    .setVerbose(1)\
    .setEvaluationLogExtended(True) \
    .setEnableOutputLogs(True)\
    .setIncludeConfidence(True)\
    .setEnableMemoryOptimizer(True)

ner_pipeline = Pipeline(stages=[
      glove_embeddings,
      nerTagger
])
```

</div>

## Results

```bash
+-----------------------------+-------+
|chunk                        |entity |
+-----------------------------+-------+
|gestational diabetes mellitus|Disease|
|diabetes mellitus            |Disease|
|T2DM                         |Disease|
|HTG-induced pancreatitis     |Disease|
|acute hepatitis              |Disease|
|obesity                      |Disease|
|polyuria                     |Disease|
|polydipsia                   |Disease|
|poor appetite                |Disease|
|vomiting                     |Disease|
+-----------------------------+-------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_ncbi_glove_100d|
|Type:|ner|
|Compatibility:|Spark NLP 3.4.0+|
|License:|Open Source|
|Edition:|Community|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|en|
|Size:|16.9 MB|
|Dependencies:|glove100d|

## Data Source

The model is trained on NCBI-disease-IOB from https://github.com/cambridgeltl/MTL-Bioinformatics-2016/tree/master/data

## Benchmarking

```bash
              precision    recall  f1-score   support

   B-Disease       0.86      0.85      0.85       960
   I-Disease       0.80      0.89      0.84      1087
           O       0.99      0.99      0.99     22450

    accuracy                           0.98     24497
   macro avg       0.88      0.91      0.90     24497
weighted avg       0.98      0.98      0.98     24497
```