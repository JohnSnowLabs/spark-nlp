---
layout: model
title: Named Entity Recognition - BERT Large (OntoNotes)
author: John Snow Labs
name: onto_bert_large_cased
date: 2020-12-05
tags: [ner, open_source, en]
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Onto is a Named Entity Recognition (or NER) model trained on OntoNotes 5.0. It can extract up to 18 entities such as people, places, organizations, money, time, date, etc.

This model uses the pretrained `bert_large_cased` model from the `BertEmbeddings` annotator as an input.

## Predicted Entities

\[`CARDINAL`, `DATE`, `EVENT`, `FAC`, `GPE`, `LANGUAGE`, `LAW`, `LOC`, `MONEY`, `NORP`, `ORDINAL`, `ORG`, `PERCENT`, `PERSON`, `PRODUCT`, `QUANTITY`, `TIME`, `WORK_OF_ART`]

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/public/NER_EN_18){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/NER_EN.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/onto_bert_large_cased_en_2.7.0_2.4_1607198127113.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPython.html %}
```python
ner = NerDLModel.pretrained("onto_bert_large_cased", "en") \
        .setInputCols(["document", "token", "embeddings"]) \
        .setOutputCol("ner")

```
```scala
val ner = NerDLModel.pretrained("onto_bert_large_cased", "en")
        .setInputCols(Array("document", "token", "embeddings"))
        .setOutputCol("ner")
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|onto_bert_large_cased|
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

prec: 0.8947816, rec: 0.9059915, f1: 0.90035164

CoNLL Eval:

processed 152728 tokens with 11257 phrases; found: 11351 phrases; correct: 10044.
accuracy:  98.02%; 10044 11257 11351 precision:  88.49%; recall:  89.22%; FB1:  88.85
         CARDINAL:  793  935  953 precision:  83.21%; recall:  84.81%; FB1:  84.00  953
             DATE:  1420  1602  1697 precision:  83.68%; recall:  88.64%; FB1:  86.09  1697
            EVENT:   37   63   63 precision:  58.73%; recall:  58.73%; FB1:  58.73  63
              FAC:   98  135  152 precision:  64.47%; recall:  72.59%; FB1:  68.29  152
              GPE:  2128  2240  2218 precision:  95.94%; recall:  95.00%; FB1:  95.47  2218
         LANGUAGE:   10   22   13 precision:  76.92%; recall:  45.45%; FB1:  57.14  13
              LAW:   21   40   30 precision:  70.00%; recall:  52.50%; FB1:  60.00  30
              LOC:  133  179  166 precision:  80.12%; recall:  74.30%; FB1:  77.10  166
            MONEY:  279  314  317 precision:  88.01%; recall:  88.85%; FB1:  88.43  317
             NORP:  796  841  840 precision:  94.76%; recall:  94.65%; FB1:  94.71  840
          ORDINAL:  180  195  219 precision:  82.19%; recall:  92.31%; FB1:  86.96  219
              ORG:  1620  1795  1873 precision:  86.49%; recall:  90.25%; FB1:  88.33  1873
          PERCENT:  309  349  342 precision:  90.35%; recall:  88.54%; FB1:  89.44  342
           PERSON:  1862  1988  1970 precision:  94.52%; recall:  93.66%; FB1:  94.09  1970
          PRODUCT:   51   76   68 precision:  75.00%; recall:  67.11%; FB1:  70.83  68
         QUANTITY:   81  105   99 precision:  81.82%; recall:  77.14%; FB1:  79.41  99
             TIME:  116  212  179 precision:  64.80%; recall:  54.72%; FB1:  59.34  179
      WORK_OF_ART:  110  166  152 precision:  72.37%; recall:  66.27%; FB1:  69.18  152

```
