---
layout: model
title: Named Entity Recognition - BERT Base (OntoNotes)
author: John Snow Labs
name: onto_bert_base_cased
date: 2020-12-05
tags: [ner, en, open_source]
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Onto is a Named Entity Recognition (or NER) model trained on OntoNotes 5.0. It can extract up to 18 entities such as people, places, organizations, money, time, date, etc.

This model uses the pretrained `bert_base_cased` model from `BertEmbeddings` annotator as an input.

## Predicted Entities

\[`CARDINAL`, `DATE`, `EVENT`, `FAC`, `GPE`, `LANGUAGE`, `LAW`, `LOC`, `MONEY`, `NORP`, `ORDINAL`, `ORG`, `PERCENT`, `PERSON`, `PRODUCT`, `QUANTITY`, `TIME`, `WORK_OF_ART`]

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/public/NER_EN_18){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/NER_EN.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/onto_bert_base_cased_en_2.7.0_2.4_1607197077494.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPython.html %}
```python
ner = NerDLModel.pretrained("onto_bert_base_cased", "en") \
        .setInputCols(["document", "token", "embeddings"]) \
        .setOutputCol("ner")

```
```scala
val ner = NerDLModel.pretrained("onto_bert_base_cased", "en")
        .setInputCols(Array("document", "token", "embeddings"))
        .setOutputCol("ner")
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|onto_bert_base_cased|
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

prec: 0.8987879, rec: 0.90063596, f1: 0.89971095
  
CoNLL Eval:

processed 152728 tokens with 11257 phrases; found: 11276 phrases; correct: 10006.
accuracy:  98.01%; 10006 11257 11276 precision:  88.74%; recall:  88.89%; FB1:  88.81
         CARDINAL:  822  935  990 precision:  83.03%; recall:  87.91%; FB1:  85.40  990
             DATE:  1355  1602  1567 precision:  86.47%; recall:  84.58%; FB1:  85.52  1567
            EVENT:   32   63   59 precision:  54.24%; recall:  50.79%; FB1:  52.46  59
              FAC:   96  135  124 precision:  77.42%; recall:  71.11%; FB1:  74.13  124
              GPE:  2116  2240  2182 precision:  96.98%; recall:  94.46%; FB1:  95.70  2182
         LANGUAGE:   10   22   11 precision:  90.91%; recall:  45.45%; FB1:  60.61  11
              LAW:   21   40   28 precision:  75.00%; recall:  52.50%; FB1:  61.76  28
              LOC:  141  179  178 precision:  79.21%; recall:  78.77%; FB1:  78.99  178
            MONEY:  278  314  321 precision:  86.60%; recall:  88.54%; FB1:  87.56  321
             NORP:  799  841  850 precision:  94.00%; recall:  95.01%; FB1:  94.50  850
          ORDINAL:  177  195  217 precision:  81.57%; recall:  90.77%; FB1:  85.92  217
              ORG:  1606  1795  1848 precision:  86.90%; recall:  89.47%; FB1:  88.17  1848
          PERCENT:  306  349  344 precision:  88.95%; recall:  87.68%; FB1:  88.31  344
           PERSON:  1856  1988  1978 precision:  93.83%; recall:  93.36%; FB1:  93.60  1978
          PRODUCT:   54   76   76 precision:  71.05%; recall:  71.05%; FB1:  71.05  76
         QUANTITY:   87  105  108 precision:  80.56%; recall:  82.86%; FB1:  81.69  108
             TIME:  143  212  216 precision:  66.20%; recall:  67.45%; FB1:  66.82  216
      WORK_OF_ART:  107  166  179 precision:  59.78%; recall:  64.46%; FB1:  62.03  179

```