---
layout: model
title: Named Entity Recognition - ELECTRA Small (OntoNotes)
author: John Snow Labs
name: onto_electra_small_uncased
date: 2020-12-05
tags: [ner, open_source, en]
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Onto is a Named Entity Recognition (or NER) model trained on OntoNotes 5.0. It can extract up to 18 entities such as people, places, organizations, money, time, date, etc.

This model uses the pretrained `electra_small_uncased` model from the `BertEmbeddings` annotator as an input.

## Predicted Entities

\[`CARDINAL`, `DATE`, `EVENT`, `FAC`, `GPE`, `LANGUAGE`, `LAW`, `LOC`, `MONEY`, `NORP`, `ORDINAL`, `ORG`, `PERCENT`, `PERSON`, `PRODUCT`, `QUANTITY`, `TIME`, `WORK_OF_ART`]

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/public/NER_EN_18){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/NER_EN.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/onto_electra_small_uncased_en_2.7.0_2.4_1607202932422.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPython.html %}
```python
ner = NerDLModel.pretrained("onto_electra_small_uncased", "en") \
        .setInputCols(["document", "token", "embeddings"]) \
        .setOutputCol("ner")
```
```scala
val ner = NerDLModel.pretrained("onto_electra_small_uncased", "en")
        .setInputCols(Array("document", "token", "embeddings"))
        .setOutputCol("ner")
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|onto_electra_small_uncased|
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

prec: 0.87234557, rec: 0.8584134, f1: 0.8653234 

CoNLL Eval:

processed 152728 tokens with 11257 phrases; found: 11149 phrases; correct: 9598.
accuracy:  97.22%; 9598 11257 11149 precision:  86.09%; recall:  85.26%; FB1:  85.67
         CARDINAL:  789  935  948 precision:  83.23%; recall:  84.39%; FB1:  83.80  948
             DATE:  1400  1602  1659 precision:  84.39%; recall:  87.39%; FB1:  85.86  1659
            EVENT:   31   63   50 precision:  62.00%; recall:  49.21%; FB1:  54.87  50
              FAC:   72  135  111 precision:  64.86%; recall:  53.33%; FB1:  58.54  111
              GPE:  2086  2240  2197 precision:  94.95%; recall:  93.12%; FB1:  94.03  2197
         LANGUAGE:    8   22   10 precision:  80.00%; recall:  36.36%; FB1:  50.00  10
              LAW:   21   40   34 precision:  61.76%; recall:  52.50%; FB1:  56.76  34
              LOC:  114  179  201 precision:  56.72%; recall:  63.69%; FB1:  60.00  201
            MONEY:  282  314  321 precision:  87.85%; recall:  89.81%; FB1:  88.82  321
             NORP:  786  841  848 precision:  92.69%; recall:  93.46%; FB1:  93.07  848
          ORDINAL:  180  195  227 precision:  79.30%; recall:  92.31%; FB1:  85.31  227
              ORG:  1359  1795  1616 precision:  84.10%; recall:  75.71%; FB1:  79.68  1616
          PERCENT:  312  349  349 precision:  89.40%; recall:  89.40%; FB1:  89.40  349
           PERSON:  1852  1988  2059 precision:  89.95%; recall:  93.16%; FB1:  91.52  2059
          PRODUCT:   32   76   69 precision:  46.38%; recall:  42.11%; FB1:  44.14  69
         QUANTITY:   86  105  105 precision:  81.90%; recall:  81.90%; FB1:  81.90  105
             TIME:  124  212  207 precision:  59.90%; recall:  58.49%; FB1:  59.19  207
      WORK_OF_ART:   64  166  138 precision:  46.38%; recall:  38.55%; FB1:  42.11  138
```