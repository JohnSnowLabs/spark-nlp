---
layout: model
title: Finance NER (10Q, md, 12 entities)
author: John Snow Labs
name: finner_10q_xlbr_lg_subset6
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

This model is an NER model containing 12 numeric financial entities from different 10Q reports. The tokens being annotated are the amounts, not any other surrounding word, but the context will determine what kind of amount is from the list of the 12 available.

This is a large (`lg`) model, trained with 200K sentences.

## Predicted Entities

`DisposalGroupIncludingDiscontinuedOperationConsideration`, `IncomeTaxExpenseBenefit`, `GainsLossesOnExtinguishmentOfDebt`, `DerivativeFixedInterestRate`, `IncomeLossFromEquityMethodInvestments`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/finance/models/finner_10q_xlbr_lg_subset6_en_1.0.0_3.0_1672638596340.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

nerTagger = finance.NerModel.pretrained('finner_10q_xlbr_lg_subset6', 'en', 'finance/models')\
   .setInputCols(["sentence", "token", "embeddings"])\
   .setOutputCol("ner")
              
pipeline = nlp.Pipeline(stages=[documentAssembler,
                            sentence,
                            tokenizer,
                            embeddings,
                            nerTagger
                                ])
text = "For the nine months ended August 31 , 2015 , Lennar Homebuilding equity in earnings included $ 64.5 million of equity in earnings from one of the Company 's unconsolidated entities primarily due to sales of approximately 700 homesites and a commercial property to third parties and a gain on debt extinguishment ."

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
|For           |O                                      |1.0       |
|the           |O                                      |1.0       |
|nine          |O                                      |1.0       |
|months        |O                                      |1.0       |
|ended         |O                                      |1.0       |
|August        |O                                      |1.0       |
|31            |O                                      |1.0       |
|,             |O                                      |1.0       |
|2015          |O                                      |1.0       |
|,             |O                                      |1.0       |
|Lennar        |O                                      |1.0       |
|Homebuilding  |O                                      |1.0       |
|equity        |O                                      |1.0       |
|in            |O                                      |1.0       |
|earnings      |O                                      |1.0       |
|included      |O                                      |1.0       |
|$             |O                                      |1.0       |
|64.5          |B-IncomeLossFromEquityMethodInvestments|0.9968    |
|million       |O                                      |1.0       |
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
|primarily     |O                                      |1.0       |
|due           |O                                      |1.0       |
|to            |O                                      |1.0       |
|sales         |O                                      |1.0       |
|of            |O                                      |1.0       |
|approximately |O                                      |1.0       |
|700           |O                                      |0.9998    |
|homesites     |O                                      |1.0       |
|and           |O                                      |1.0       |
|a             |O                                      |1.0       |
|commercial    |O                                      |1.0       |
|property      |O                                      |1.0       |
|to            |O                                      |1.0       |
|third         |O                                      |1.0       |
|parties       |O                                      |1.0       |
|and           |O                                      |1.0       |
|a             |O                                      |1.0       |
|gain          |O                                      |1.0       |
|on            |O                                      |1.0       |
|debt          |O                                      |1.0       |
|extinguishment|O                                      |1.0       |
|.             |O                                      |1.0       |
+--------------+---------------------------------------+----------+


```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|finner_10q_xlbr_lg_subset6|
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
                                                 macro avg     0.9791    0.9810    0.9800     44933
                                              weighted avg     0.9986    0.9986    0.9986     44933

```