---
layout: model
title: Finance Income NER (10Q, lg)
author: John Snow Labs
name: finner_10q_xlbr_lg_income
date: 2023-01-02
tags: [10q, xlbr, en, licensed]
task: Named Entity Recognition
language: en
edition: Finance NLP 1.0.0
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model is an NER model containing 5 numeric financial Income entities from different 10Q reports. The tokens being annotated are the amounts, not any other surrounding word, but the context will determine what kind of amount is from the list of the 5 available.

This is a large (`lg`) model, trained with 200K sentences.

## Predicted Entities

`IncomeLossFromEquityMethodInvestments`, `DisposalGroupIncludingDiscontinuedOperationConsideration`, `GainsLossesOnExtinguishmentOfDebt`, `DerivativeFixedInterestRate`, `IncomeTaxExpenseBenefit`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/finance/models/finner_10q_xlbr_lg_income_en_1.0.0_3.0_1672653711790.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
 
documentAssembler = nlp.DocumentAssembler() \
   .setInputCol("text") \
   .setOutputCol("document")

sentence = nlp.SentenceDetector() \
   .setInputCols(["document"]) \
   .setOutputCol("sentence") 

tokenizer = nlp.Tokenizer()\
    .setInputCols(["sentence"])\
    .setOutputCol("token")\
    .setContextChars(['.', ',', ';', ':', '!', '?', '*', '-', '(', ')', '”', '’', '$','€'])

embeddings = nlp.BertEmbeddings.pretrained("bert_embeddings_sec_bert_base","en") \
  .setInputCols(["document", "token"]) \
  .setOutputCol("embeddings")\
  .setMaxSentenceLength(512)

nerTagger = finance.NerModel.pretrained('finner_10q_xlbr_lg_income', 'en', 'finance/models')\
   .setInputCols(["sentence", "token", "embeddings"])\
   .setOutputCol("ner")
              
pipeline = nlp.Pipeline(stages=[documentAssembler,
                            sentence,
                            tokenizer,
                            embeddings,
                            nerTagger
                                ])
text = "This was partially offset by $ 6.7 million and $ 12.7 million , respectively , of equity in earnings from one of the Company 's unconsolidated entities for the three and six months ended May 31 , 2016 primarily due to sales of 253 homesites and 471 homesites , respectively , to third parties for $ 52.1 million and $ 114.1 million , respectively , that resulted in gross profits of $ 18.3 million and $ 39.0 million , respectively ."

df = spark.createDataFrame([[text]]).toDF("text")
fit = pipeline.fit(df)

result = fit.transform(df)

result_df = result.select(F.explode(F.arrays_zip(result.token.result,result.ner.result, result.ner.metadata)).alias("cols"))\
.select(F.expr("cols['0']").alias("token"),\
      F.expr("cols['1']").alias("ner_label"),\
      F.expr("cols['2']['confidence']").alias("confidence"))

result_df.show(50, truncate=100)
```

</div>

## Results

```bash

+--------------+---------------------------------------+----------+
|token         |ner_label                              |confidence|
+--------------+---------------------------------------+----------+
|This          |O                                      |1.0       |
|was           |O                                      |1.0       |
|partially     |O                                      |1.0       |
|offset        |O                                      |1.0       |
|by            |O                                      |1.0       |
|$             |O                                      |1.0       |
|6.7           |B-IncomeLossFromEquityMethodInvestments|0.9998    |
|million       |O                                      |1.0       |
|and           |O                                      |1.0       |
|$             |O                                      |1.0       |
|12.7          |B-IncomeLossFromEquityMethodInvestments|0.9975    |
|million       |O                                      |1.0       |
|,             |O                                      |1.0       |
|respectively  |O                                      |1.0       |
|,             |O                                      |1.0       |
|of            |O                                      |1.0       |
|equity        |O                                      |1.0       |
|in            |O                                      |1.0       |
|earnings      |O                                      |1.0       |
|from          |O                                      |1.0       |
|one           |O                                      |1.0       |
|of            |O                                      |1.0       |
|the           |O                                      |1.0       |
|Company       |O                                      |1.0       |
|'s            |O                                      |1.0       |
|unconsolidated|O                                      |1.0       |
|entities      |O                                      |1.0       |
|for           |O                                      |1.0       |
|the           |O                                      |1.0       |
|three         |O                                      |1.0       |
|and           |O                                      |1.0       |
|six           |O                                      |1.0       |
|months        |O                                      |1.0       |
|ended         |O                                      |1.0       |
|May           |O                                      |1.0       |
|31            |O                                      |1.0       |
|,             |O                                      |1.0       |
|2016          |O                                      |1.0       |
|primarily     |O                                      |1.0       |
|due           |O                                      |1.0       |
|to            |O                                      |1.0       |
|sales         |O                                      |1.0       |
|of            |O                                      |1.0       |
|253           |O                                      |0.9997    |
|homesites     |O                                      |1.0       |
|and           |O                                      |1.0       |
|471           |O                                      |0.9999    |
|homesites     |O                                      |1.0       |
|,             |O                                      |1.0       |
|respectively  |O                                      |1.0       |
|,             |O                                      |1.0       |
|to            |O                                      |1.0       |
|third         |O                                      |1.0       |
|parties       |O                                      |1.0       |
|for           |O                                      |1.0       |
|$             |O                                      |1.0       |
|52.1          |O                                      |0.9845    |
|million       |O                                      |1.0       |
|and           |O                                      |1.0       |
|$             |O                                      |1.0       |
|114.1         |O                                      |0.9984    |
|million       |O                                      |1.0       |
|,             |O                                      |1.0       |
|respectively  |O                                      |1.0       |
|,             |O                                      |1.0       |
|that          |O                                      |1.0       |
|resulted      |O                                      |1.0       |
|in            |O                                      |1.0       |
|gross         |O                                      |1.0       |
|profits       |O                                      |1.0       |
|of            |O                                      |1.0       |
|$             |O                                      |1.0       |
|18.3          |O                                      |0.7789    |
|million       |O                                      |1.0       |
|and           |O                                      |1.0       |
|$             |O                                      |1.0       |
|39.0          |O                                      |0.9363    |
|million       |O                                      |1.0       |
|,             |O                                      |1.0       |
|respectively  |O                                      |1.0       |
|.             |O                                      |1.0       |
+--------------+---------------------------------------+----------+

```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|finner_10q_xlbr_lg_income|
|Compatibility:|Finance NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|en|
|Size:|16.3 MB|

## References

An in-house modified version of https://huggingface.co/datasets/nlpaueb/finer-139, re-splited and filtered to focus on sentences with bigger density of tags.

## Benchmarking

```bash


label                                                        precision    recall  f1-score   support
                             B-DerivativeFixedInterestRate     0.9858    1.0000    0.9929       139
B-DisposalGroupIncludingDiscontinuedOperationConsideration     0.9756    0.9479    0.9615       211
                       B-GainsLossesOnExtinguishmentOfDebt     0.9502    0.9709    0.9604       275
                   B-IncomeLossFromEquityMethodInvestments     0.9734    0.9846    0.9790       260
                                 B-IncomeTaxExpenseBenefit     0.9901    0.9832    0.9866       713
                                                         O     0.9993    0.9993    0.9993     43335
                                                  accuracy         -         -     0.9986     44933
                                                 macro-avg     0.9791    0.9810    0.9800     44933
                                              weighted-avg     0.9986    0.9986    0.9986     44933

```
