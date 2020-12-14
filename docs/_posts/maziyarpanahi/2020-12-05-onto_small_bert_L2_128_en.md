---
layout: model
title: Named Entity Recognition - BERT Tiny (OntoNotes)
author: John Snow Labs
name: onto_small_bert_L2_128
date: 2020-12-05
tags: [ner, open_source, en]
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Onto is a Named Entity Recognition (or NER) model trained on OntoNotes 5.0. It can extract up to 18 entities such as people, places, organizations, money, time, date, etc.

This model uses the pretrained `small_bert_L2_128` model from the `BertEmbeddings` annotator as an input.

## Predicted Entities

\[`CARDINAL`, `DATE`, `EVENT`, `FAC`, `GPE`, `LANGUAGE`, `LAW`, `LOC`, `MONEY`, `NORP`, `ORDINAL`, `ORG`, `PERCENT`, `PERSON`, `PRODUCT`, `QUANTITY`, `TIME`, `WORK_OF_ART`]

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/public/NER_EN_18){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/NER_EN.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/onto_small_bert_L2_128_en_2.7.0_2.4_1607198998042.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPython.html %}
```python
ner = NerDLModel.pretrained("onto_small_bert_L2_128", "en") \
        .setInputCols(["document", "token", "embeddings"]) \
        .setOutputCol("ner")
```
```scala
val ner = NerDLModel.pretrained("onto_small_bert_L2_128", "en")
        .setInputCols(Array("document", "token", "embeddings"))
        .setOutputCol("ner")
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|onto_small_bert_L2_128|
|Type:|ner|
|Compatibility:|Spark NLP 2.7.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|en|

## Data Source

The model is trained based on data from OntoNotes 5.0 [https://catalog.ldc.upenn.edu/LDC2013T19](https://catalog.ldc.upenn.edu/LDC2013T19)

## Benchmarking

```bash
Micro-average:

prec: 0.86477494, rec: 0.8204466, f1: 0.8420278

CoNLL Eval:
  
processed 152728 tokens with 11257 phrases; found: 10772 phrases; correct: 9153.
accuracy:  96.73%; 9153 11257 10772 precision:  84.97%; recall:  81.31%; FB1:  83.10
         CARDINAL:  733  935  890 precision:  82.36%; recall:  78.40%; FB1:  80.33  890
             DATE:  1278  1602  1494 precision:  85.54%; recall:  79.78%; FB1:  82.56  1494
            EVENT:   22   63   45 precision:  48.89%; recall:  34.92%; FB1:  40.74  45
              FAC:   67  135  114 precision:  58.77%; recall:  49.63%; FB1:  53.82  114
              GPE:  2044  2240  2201 precision:  92.87%; recall:  91.25%; FB1:  92.05  2201
         LANGUAGE:    8   22   14 precision:  57.14%; recall:  36.36%; FB1:  44.44  14
              LAW:   12   40   15 precision:  80.00%; recall:  30.00%; FB1:  43.64  15
              LOC:  104  179  155 precision:  67.10%; recall:  58.10%; FB1:  62.28  155
            MONEY:  265  314  316 precision:  83.86%; recall:  84.39%; FB1:  84.13  316
             NORP:  775  841  886 precision:  87.47%; recall:  92.15%; FB1:  89.75  886
          ORDINAL:  180  195  239 precision:  75.31%; recall:  92.31%; FB1:  82.95  239
              ORG:  1280  1795  1548 precision:  82.69%; recall:  71.31%; FB1:  76.58  1548
          PERCENT:  308  349  350 precision:  88.00%; recall:  88.25%; FB1:  88.13  350
           PERSON:  1784  1988  2032 precision:  87.80%; recall:  89.74%; FB1:  88.76  2032
          PRODUCT:   33   76   49 precision:  67.35%; recall:  43.42%; FB1:  52.80  49
         QUANTITY:   83  105  112 precision:  74.11%; recall:  79.05%; FB1:  76.50  112
             TIME:  124  212  205 precision:  60.49%; recall:  58.49%; FB1:  59.47  205
      WORK_OF_ART:   53  166  107 precision:  49.53%; recall:  31.93%; FB1:  38.83  107
```