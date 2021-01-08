---
layout: model
title: Named Entity Recognition - BERT Mini (OntoNotes)
author: John Snow Labs
name: onto_small_bert_L4_256
date: 2020-12-05
tags: [ner, open_source, en]
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Onto is a Named Entity Recognition (or NER) model trained on OntoNotes 5.0. It can extract up to 18 entities such as people, places, organizations, money, time, date, etc.

This model uses the pretrained `small_bert_L4_256` model from the `BertEmbeddings` annotator as an input.

## Predicted Entities

\[`CARDINAL`, `DATE`, `EVENT`, `FAC`, `GPE`, `LANGUAGE`, `LAW`, `LOC`, `MONEY`, `NORP`, `ORDINAL`, `ORG`, `PERCENT`, `PERSON`, `PRODUCT`, `QUANTITY`, `TIME`, `WORK_OF_ART`]

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/public/NER_EN_18){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/NER_EN.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/onto_small_bert_L4_256_en_2.7.0_2.4_1607199231735.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPython.html %}
```python
ner = NerDLModel.pretrained("onto_small_bert_L4_256", "en") \
        .setInputCols(["document", "token", "embeddings"]) \
        .setOutputCol("ner")
```
```scala
val ner = NerDLModel.pretrained("onto_small_bert_L4_256", "en")
        .setInputCols(Array("document", "token", "embeddings"))
        .setOutputCol("ner")
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|onto_small_bert_L4_256|
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

prec: 0.8617996, rec: 0.85458803, f1: 0.8581787

CoNLL Eval:

processed 152728 tokens with 11257 phrases; found: 11191 phrases; correct: 9476.
accuracy:  97.05%; 9476 11257 11191 precision:  84.68%; recall:  84.18%; FB1:  84.43
         CARDINAL:  771  935  934 precision:  82.55%; recall:  82.46%; FB1:  82.50  934
             DATE:  1383  1602  1645 precision:  84.07%; recall:  86.33%; FB1:  85.19  1645
            EVENT:   29   63   49 precision:  59.18%; recall:  46.03%; FB1:  51.79  49
              FAC:   65  135  100 precision:  65.00%; recall:  48.15%; FB1:  55.32  100
              GPE:  2054  2240  2211 precision:  92.90%; recall:  91.70%; FB1:  92.29  2211
         LANGUAGE:   10   22   13 precision:  76.92%; recall:  45.45%; FB1:  57.14  13
              LAW:   11   40   22 precision:  50.00%; recall:  27.50%; FB1:  35.48  22
              LOC:  112  179  186 precision:  60.22%; recall:  62.57%; FB1:  61.37  186
            MONEY:  272  314  317 precision:  85.80%; recall:  86.62%; FB1:  86.21  317
             NORP:  781  841  856 precision:  91.24%; recall:  92.87%; FB1:  92.04  856
          ORDINAL:  172  195  228 precision:  75.44%; recall:  88.21%; FB1:  81.32  228
              ORG:  1383  1795  1749 precision:  79.07%; recall:  77.05%; FB1:  78.05  1749
          PERCENT:  311  349  346 precision:  89.88%; recall:  89.11%; FB1:  89.50  346
           PERSON:  1809  1988  2048 precision:  88.33%; recall:  91.00%; FB1:  89.64  2048
          PRODUCT:   34   76   50 precision:  68.00%; recall:  44.74%; FB1:  53.97  50
         QUANTITY:   83  105  106 precision:  78.30%; recall:  79.05%; FB1:  78.67  106
             TIME:  138  212  228 precision:  60.53%; recall:  65.09%; FB1:  62.73  228
      WORK_OF_ART:   58  166  103 precision:  56.31%; recall:  34.94%; FB1:  43.12  103      
```