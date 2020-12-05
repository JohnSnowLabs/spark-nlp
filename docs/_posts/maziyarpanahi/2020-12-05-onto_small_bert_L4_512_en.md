---
layout: model
title: Named Entity Recognition - BERT Small (OntoNotes)
author: John Snow Labs
name: onto_small_bert_L4_512
date: 2020-12-05
tags: [ner, open_source, en]
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Onto is a Named Entity Recognition (or NER) model trained on OntoNotes 5.0. It can extract up to 18 entities such as people, places, organizations, money, time, date, etc.

This model uses the pretrained `small_bert_L4_512` model from the `BertEmbeddings` annotator as an input.

## Predicted Entities

\[`CARDINAL`, `DATE`, `EVENT`, `FAC`, `GPE`, `LANGUAGE`, `LAW`, `LOC`, `MONEY`, `NORP`, `ORDINAL`, `ORG`, `PERCENT`, `PERSON`, `PRODUCT`, `QUANTITY`, `TIME`, `WORK_OF_ART`]

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/public/NER_EN_18){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/NER_EN.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/onto_small_bert_L4_512_en_2.7.0_2.4_1607199400149.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPython.html %}
```python
ner = NerDLModel.pretrained("onto_small_bert_L4_512", "en") \
        .setInputCols(["document", "token", "embeddings"]) \
        .setOutputCol("ner")
```
```scala
val ner = NerDLModel.pretrained("onto_small_bert_L4_512", "en")
        .setInputCols(Array("document", "token", "embeddings"))
        .setOutputCol("ner")
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|onto_small_bert_L4_512|
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

prec: 0.8697573, rec: 0.8567398, f1: 0.8631994

CoNLL Eval:

processed 152728 tokens with 11257 phrases; found: 11219 phrases; correct: 9557.
accuracy:  97.17%; 9557 11257 11219 precision:  85.19%; recall:  84.90%; FB1:  85.04
         CARDINAL:  804  935  958 precision:  83.92%; recall:  85.99%; FB1:  84.94  958
             DATE:  1412  1602  1726 precision:  81.81%; recall:  88.14%; FB1:  84.86  1726
            EVENT:   20   63   46 precision:  43.48%; recall:  31.75%; FB1:  36.70  46
              FAC:   78  135  122 precision:  63.93%; recall:  57.78%; FB1:  60.70  122
              GPE:  2066  2240  2185 precision:  94.55%; recall:  92.23%; FB1:  93.38  2185
         LANGUAGE:   10   22   11 precision:  90.91%; recall:  45.45%; FB1:  60.61  11
              LAW:   12   40   18 precision:  66.67%; recall:  30.00%; FB1:  41.38  18
              LOC:  114  179  168 precision:  67.86%; recall:  63.69%; FB1:  65.71  168
            MONEY:  273  314  320 precision:  85.31%; recall:  86.94%; FB1:  86.12  320
             NORP:  779  841  873 precision:  89.23%; recall:  92.63%; FB1:  90.90  873
          ORDINAL:  174  195  226 precision:  76.99%; recall:  89.23%; FB1:  82.66  226
              ORG:  1381  1795  1691 precision:  81.67%; recall:  76.94%; FB1:  79.23  1691
          PERCENT:  311  349  349 precision:  89.11%; recall:  89.11%; FB1:  89.11  349
           PERSON:  1827  1988  2046 precision:  89.30%; recall:  91.90%; FB1:  90.58  2046
          PRODUCT:   32   76   51 precision:  62.75%; recall:  42.11%; FB1:  50.39  51
         QUANTITY:   80  105  105 precision:  76.19%; recall:  76.19%; FB1:  76.19  105
             TIME:  124  212  219 precision:  56.62%; recall:  58.49%; FB1:  57.54  219
      WORK_OF_ART:   60  166  105 precision:  57.14%; recall:  36.14%; FB1:  44.28  105
```