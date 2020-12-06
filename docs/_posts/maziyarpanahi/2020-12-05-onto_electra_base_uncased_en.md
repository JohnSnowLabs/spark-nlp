---
layout: model
title: Named Entity Recognition - ELECTRA Base (OntoNotes)
author: John Snow Labs
name: onto_electra_base_uncased
date: 2020-12-05
tags: [ner, open_source, en]
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Onto is a Named Entity Recognition (or NER) model trained on OntoNotes 5.0. It can extract up to 18 entities such as people, places, organizations, money, time, date, etc.

This model uses the pretrained `electra_base_uncased` model from the `BertEmbeddings` annotator as an input.

## Predicted Entities

\[`CARDINAL`, `DATE`, `EVENT`, `FAC`, `GPE`, `LANGUAGE`, `LAW`, `LOC`, `MONEY`, `NORP`, `ORDINAL`, `ORG`, `PERCENT`, `PERSON`, `PRODUCT`, `QUANTITY`, `TIME`, `WORK_OF_ART`]

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/public/NER_EN_18){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/NER_EN.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/onto_electra_base_uncased_en_2.7.0_2.4_1607203076517.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPython.html %}
```python
ner = NerDLModel.pretrained("onto_electra_base_uncased", "en") \
        .setInputCols(["document", "token", "embeddings"]) \
        .setOutputCol("ner")
```
```scala
val ner = NerDLModel.pretrained("onto_electra_base_uncased", "en")
        .setInputCols(Array("document", "token", "embeddings"))
        .setOutputCol("ner")
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|onto_electra_base_uncased|
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

prec: 0.88154626, rec: 0.88217854, f1: 0.8818623

CoNLL Eval:

processed 152728 tokens with 11257 phrases; found: 11296 phrases; correct: 9871.
accuracy:  97.56%; 9871 11257 11296 precision:  87.38%; recall:  87.69%; FB1:  87.54
         CARDINAL:  800  935  959 precision:  83.42%; recall:  85.56%; FB1:  84.48  959
             DATE:  1396  1602  1652 precision:  84.50%; recall:  87.14%; FB1:  85.80  1652
            EVENT:   23   63   38 precision:  60.53%; recall:  36.51%; FB1:  45.54  38
              FAC:   60  135   81 precision:  74.07%; recall:  44.44%; FB1:  55.56  81
              GPE:  2102  2240  2205 precision:  95.33%; recall:  93.84%; FB1:  94.58  2205
         LANGUAGE:    8   22   10 precision:  80.00%; recall:  36.36%; FB1:  50.00  10
              LAW:   16   40   21 precision:  76.19%; recall:  40.00%; FB1:  52.46  21
              LOC:  118  179  191 precision:  61.78%; recall:  65.92%; FB1:  63.78  191
            MONEY:  285  314  329 precision:  86.63%; recall:  90.76%; FB1:  88.65  329
             NORP:  801  841  897 precision:  89.30%; recall:  95.24%; FB1:  92.17  897
          ORDINAL:  180  195  225 precision:  80.00%; recall:  92.31%; FB1:  85.71  225
              ORG:  1538  1795  1816 precision:  84.69%; recall:  85.68%; FB1:  85.18  1816
          PERCENT:  312  349  348 precision:  89.66%; recall:  89.40%; FB1:  89.53  348
           PERSON:  1892  1988  2025 precision:  93.43%; recall:  95.17%; FB1:  94.29  2025
          PRODUCT:   39   76   49 precision:  79.59%; recall:  51.32%; FB1:  62.40  49
         QUANTITY:   81  105  102 precision:  79.41%; recall:  77.14%; FB1:  78.26  102
             TIME:  136  212  216 precision:  62.96%; recall:  64.15%; FB1:  63.55  216
      WORK_OF_ART:   84  166  132 precision:  63.64%; recall:  50.60%; FB1:  56.38  132
```