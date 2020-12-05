---
layout: model
title: Named Entity Recognition - BERT Medium (OntoNotes)
author: John Snow Labs
name: onto_small_bert_L8_512
date: 2020-12-05
tags: [ner, open_source, en]
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Onto is a Named Entity Recognition (or NER) model trained on OntoNotes 5.0. It can extract up to 18 entities such as people, places, organizations, money, time, date, etc.

This model uses the pretrained `small_bert_L8_512` model from the `BertEmbeddings` annotator as an input.

## Predicted Entities

\[`CARDINAL`, `DATE`, `EVENT`, `FAC`, `GPE`, `LANGUAGE`, `LAW`, `LOC`, `MONEY`, `NORP`, `ORDINAL`, `ORG`, `PERCENT`, `PERSON`, `PRODUCT`, `QUANTITY`, `TIME`, `WORK_OF_ART`]

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/public/NER_EN_18){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/NER_EN.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/onto_small_bert_L8_512_en_2.7.0_2.4_1607199531477.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPython.html %}
```python
ner = NerDLModel.pretrained("onto_small_bert_L8_512", "en") \
        .setInputCols(["document", "token", "embeddings"]) \
        .setOutputCol("ner")
```
```scala
val ner = NerDLModel.pretrained("onto_small_bert_L8_512", "en")
        .setInputCols(Array("document", "token", "embeddings"))
        .setOutputCol("ner")
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|onto_small_bert_L8_512|
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

prec: 0.8849518, rec: 0.85147995, f1: 0.8678933

CoNLL Eval:

processed 152728 tokens with 11257 phrases; found: 11073 phrases; correct: 9556.
accuracy:  97.26%; 9556 11257 11073 precision:  86.30%; recall:  84.89%; FB1:  85.59
         CARDINAL:  798  935  929 precision:  85.90%; recall:  85.35%; FB1:  85.62  929
             DATE:  1410  1602  1654 precision:  85.25%; recall:  88.01%; FB1:  86.61  1654
            EVENT:   23   63   44 precision:  52.27%; recall:  36.51%; FB1:  42.99  44
              FAC:   79  135  121 precision:  65.29%; recall:  58.52%; FB1:  61.72  121
              GPE:  2097  2240  2244 precision:  93.45%; recall:  93.62%; FB1:  93.53  2244
         LANGUAGE:    9   22   11 precision:  81.82%; recall:  40.91%; FB1:  54.55  11
              LAW:   14   40   20 precision:  70.00%; recall:  35.00%; FB1:  46.67  20
              LOC:  111  179  152 precision:  73.03%; recall:  62.01%; FB1:  67.07  152
            MONEY:  282  314  320 precision:  88.12%; recall:  89.81%; FB1:  88.96  320
             NORP:  755  841  889 precision:  84.93%; recall:  89.77%; FB1:  87.28  889
          ORDINAL:  169  195  201 precision:  84.08%; recall:  86.67%; FB1:  85.35  201
              ORG:  1368  1795  1624 precision:  84.24%; recall:  76.21%; FB1:  80.02  1624
          PERCENT:  309  349  351 precision:  88.03%; recall:  88.54%; FB1:  88.29  351
           PERSON:  1816  1988  2037 precision:  89.15%; recall:  91.35%; FB1:  90.24  2037
          PRODUCT:   42   76   67 precision:  62.69%; recall:  55.26%; FB1:  58.74  67
         QUANTITY:   85  105  108 precision:  78.70%; recall:  80.95%; FB1:  79.81  108
             TIME:  137  212  222 precision:  61.71%; recall:  64.62%; FB1:  63.13  222
      WORK_OF_ART:   52  166   79 precision:  65.82%; recall:  31.33%; FB1:  42.45  79
```