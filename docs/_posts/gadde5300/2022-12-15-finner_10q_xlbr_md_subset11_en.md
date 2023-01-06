---
layout: model
title: Finance NER (10Q, md, 12 entities)
author: John Snow Labs
name: finner_10q_xlbr_md_subset11
date: 2022-12-15
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

This is a large (`md`) model, trained with 200K sentences.

## Predicted Entities

`SharePrice`, `SharebasedCompensationArrangementBySharebasedPaymentAwardAwardVestingRightsPercentage`, `StockRepurchaseProgramRemainingAuthorizedRepurchaseAmount1`, `ShareBasedCompensationArrangementByShareBasedPaymentAwardOptionsGrantsInPeriodWeightedAverageGrantDateFairValue`, `ShareBasedCompensationArrangementByShareBasedPaymentAwardNumberOfSharesAuthorized`, `ShareBasedCompensationArrangementByShareBasedPaymentAwardNumberOfSharesAvailableForGrant`, `SharebasedCompensationArrangementBySharebasedPaymentAwardExpirationPeriod`, `ShareBasedCompensationArrangementByShareBasedPaymentAwardOptionsGrantsInPeriodGross`, `StockRepurchasedAndRetiredDuringPeriodShares`, `StockRepurchaseProgramAuthorizedAmount1`, `ShareBasedCompensationArrangementByShareBasedPaymentAwardOptionsExercisesInPeriodTotalIntrinsicValue`, `StockIssuedDuringPeriodSharesNewIssues`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/finance/models/finner_10q_xlbr_md_subset11_en_1.0.0_3.0_1671083155687.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

nerTagger = finance.NerModel.pretrained('finner_10q_xlbr_md_subset11', 'en', 'finance/models')\
   .setInputCols(["sentence", "token", "embeddings"])\
   .setOutputCol("ner")
              
pipeline = nlp.Pipeline(stages=[documentAssembler,
                            sentence,
                            tokenizer,
                            embeddings,
                            nerTagger
                                ])
text = "The fair value of the stock option grants below were estimated on the date of the grant using a Black - Scholes valuation model and the assumptions in the following table : On December 1 , 2015 , the Company granted non - qualifed stock options under the Plan for 75,000 shares each to four directors : Sardar Biglari , Philip Cooley , Christopher Hogg and S."

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

+-----------+-------------------------------------------------------------------------------------+----------+
|token      |ner_label                                                                            |confidence|
+-----------+-------------------------------------------------------------------------------------+----------+
|The        |O                                                                                    |1.0       |
|fair       |O                                                                                    |1.0       |
|value      |O                                                                                    |1.0       |
|of         |O                                                                                    |1.0       |
|the        |O                                                                                    |1.0       |
|stock      |O                                                                                    |1.0       |
|option     |O                                                                                    |1.0       |
|grants     |O                                                                                    |1.0       |
|below      |O                                                                                    |1.0       |
|were       |O                                                                                    |1.0       |
|estimated  |O                                                                                    |1.0       |
|on         |O                                                                                    |1.0       |
|the        |O                                                                                    |1.0       |
|date       |O                                                                                    |1.0       |
|of         |O                                                                                    |1.0       |
|the        |O                                                                                    |1.0       |
|grant      |O                                                                                    |1.0       |
|using      |O                                                                                    |1.0       |
|a          |O                                                                                    |1.0       |
|Black      |O                                                                                    |1.0       |
|-          |O                                                                                    |1.0       |
|Scholes    |O                                                                                    |1.0       |
|valuation  |O                                                                                    |1.0       |
|model      |O                                                                                    |1.0       |
|and        |O                                                                                    |1.0       |
|the        |O                                                                                    |1.0       |
|assumptions|O                                                                                    |1.0       |
|in         |O                                                                                    |1.0       |
|the        |O                                                                                    |1.0       |
|following  |O                                                                                    |1.0       |
|table      |O                                                                                    |1.0       |
|:          |O                                                                                    |1.0       |
|On         |O                                                                                    |1.0       |
|December   |O                                                                                    |1.0       |
|1          |O                                                                                    |1.0       |
|,          |O                                                                                    |1.0       |
|2015       |O                                                                                    |1.0       |
|,          |O                                                                                    |1.0       |
|the        |O                                                                                    |1.0       |
|Company    |O                                                                                    |1.0       |
|granted    |O                                                                                    |1.0       |
|non        |O                                                                                    |0.9995    |
|-          |O                                                                                    |1.0       |
|qualifed   |O                                                                                    |1.0       |
|stock      |O                                                                                    |1.0       |
|options    |O                                                                                    |1.0       |
|under      |O                                                                                    |1.0       |
|the        |O                                                                                    |1.0       |
|Plan       |O                                                                                    |1.0       |
|for        |O                                                                                    |1.0       |
|75,000     |B-ShareBasedCompensationArrangementByShareBasedPaymentAwardOptionsGrantsInPeriodGross|0.9989    |
|shares     |O                                                                                    |1.0       |
|each       |O                                                                                    |1.0       |
|to         |O                                                                                    |1.0       |
|four       |O                                                                                    |1.0       |
|directors  |O                                                                                    |1.0       |
|:          |O                                                                                    |1.0       |
|Sardar     |O                                                                                    |1.0       |
|Biglari    |O                                                                                    |1.0       |
|,          |O                                                                                    |1.0       |
|Philip     |O                                                                                    |1.0       |
|Cooley     |O                                                                                    |1.0       |
|,          |O                                                                                    |1.0       |
|Christopher|O                                                                                    |1.0       |
|Hogg       |O                                                                                    |1.0       |
|and        |O                                                                                    |1.0       |
|S          |O                                                                                    |1.0       |
|.          |O                                                                                    |1.0       |
+-----------+-------------------------------------------------------------------------------------+----------+

```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|finner_10q_xlbr_md_subset11|
|Compatibility:|Finance NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|en|
|Size:|16.5 MB|

## References

An in-house modified version of https://huggingface.co/datasets/nlpaueb/finer-139, re-splited and filtered to focus on sentences with bigger density of tags.