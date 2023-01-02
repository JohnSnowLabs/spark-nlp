---
layout: model
title: Finance NER (10Q, md, 12 entities)
author: John Snow Labs
name: finner_10q_xlbr_lg_subset8
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

`SaleOfStockNumberOfSharesIssuedInTransaction`, `AntidilutiveSecuritiesExcludedFromComputationOfEarningsPerShareAmount`, `CommonStockSharesOutstanding`, `SharePrice`, `ProceedsFromIssuanceOfCommonStock`, `SaleOfStockPricePerShare`, `CommonStockParOrStatedValuePerShare`, `BusinessAcquisitionEquityInterestsIssuedOrIssuableNumberOfSharesIssued`, `CommonStockCapitalSharesReservedForFutureIssuance`, `StockIssuedDuringPeriodSharesNewIssues`, `CommonStockSharesAuthorized`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/finance/models/finner_10q_xlbr_lg_subset8_en_1.0.0_3.0_1672638901231.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

nerTagger = finance.NerModel.pretrained('finner_10q_xlbr_lg_subset8', 'en', 'finance/models')\
   .setInputCols(["sentence", "token", "embeddings"])\
   .setOutputCol("ner")
              
pipeline = nlp.Pipeline(stages=[documentAssembler,
                            sentence,
                            tokenizer,
                            embeddings,
                            nerTagger
                                ])
text = "For the three and six month periods ending May 31 , 2016 and 2015 there were 404,000 and 312,000 shares , respectively , underlying previously issued stock options that were excluded from diluted loss per share because the effects of such shares were anti - dilutive ."

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

+------------+-----------------------------------------------------------------------+----------+
|token       |ner_label                                                              |confidence|
+------------+-----------------------------------------------------------------------+----------+
|For         |O                                                                      |1.0       |
|the         |O                                                                      |1.0       |
|three       |O                                                                      |1.0       |
|and         |O                                                                      |1.0       |
|six         |O                                                                      |1.0       |
|month       |O                                                                      |1.0       |
|periods     |O                                                                      |1.0       |
|ending      |O                                                                      |1.0       |
|May         |O                                                                      |1.0       |
|31          |O                                                                      |1.0       |
|,           |O                                                                      |1.0       |
|2016        |O                                                                      |1.0       |
|and         |O                                                                      |1.0       |
|2015        |O                                                                      |1.0       |
|there       |O                                                                      |1.0       |
|were        |O                                                                      |1.0       |
|404,000     |B-AntidilutiveSecuritiesExcludedFromComputationOfEarningsPerShareAmount|0.9999    |
|and         |O                                                                      |1.0       |
|312,000     |B-AntidilutiveSecuritiesExcludedFromComputationOfEarningsPerShareAmount|1.0       |
|shares      |O                                                                      |1.0       |
|,           |O                                                                      |1.0       |
|respectively|O                                                                      |1.0       |
|,           |O                                                                      |1.0       |
|underlying  |O                                                                      |1.0       |
|previously  |O                                                                      |1.0       |
|issued      |O                                                                      |1.0       |
|stock       |O                                                                      |1.0       |
|options     |O                                                                      |1.0       |
|that        |O                                                                      |1.0       |
|were        |O                                                                      |1.0       |
|excluded    |O                                                                      |1.0       |
|from        |O                                                                      |1.0       |
|diluted     |O                                                                      |1.0       |
|loss        |O                                                                      |1.0       |
|per         |O                                                                      |1.0       |
|share       |O                                                                      |1.0       |
|because     |O                                                                      |1.0       |
|the         |O                                                                      |1.0       |
|effects     |O                                                                      |1.0       |
|of          |O                                                                      |1.0       |
|such        |O                                                                      |1.0       |
|shares      |O                                                                      |1.0       |
|were        |O                                                                      |1.0       |
|anti        |O                                                                      |1.0       |
|-           |O                                                                      |1.0       |
|dilutive    |O                                                                      |1.0       |
|.           |O                                                                      |1.0       |
+------------+-----------------------------------------------------------------------+----------+

```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|finner_10q_xlbr_lg_subset8|
|Compatibility:|Finance NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|en|
|Size:|16.4 MB|

## References

An in-house modified version of https://huggingface.co/datasets/nlpaueb/finer-139, re-splited and filtered to focus on sentences with bigger density of tags.

## Benchmarking

```bash

label                                                                       precision    recall  f1-score   support
 B-AntidilutiveSecuritiesExcludedFromComputationOfEarningsPerShareAmount     0.9913    0.9933    0.9923      1487
B-BusinessAcquisitionEquityInterestsIssuedOrIssuableNumberOfSharesIssued     0.8814    0.8062    0.8421       129
                     B-CommonStockCapitalSharesReservedForFutureIssuance     0.9515    0.9290    0.9401       169
                                   B-CommonStockParOrStatedValuePerShare     0.9249    0.9467    0.9357       169
                                           B-CommonStockSharesAuthorized     0.9500    0.9301    0.9399       143
                                          B-CommonStockSharesOutstanding     0.8443    0.9463    0.8924       149
                                     B-ProceedsFromIssuanceOfCommonStock     0.7550    0.8444    0.7972       135
                          B-SaleOfStockNumberOfSharesIssuedInTransaction     0.4486    0.8836    0.5951       232
                                              B-SaleOfStockPricePerShare     0.5774    0.9262    0.7113       149
                                                            B-SharePrice     0.9338    0.7056    0.8038       180
                                B-StockIssuedDuringPeriodSharesNewIssues     0.7725    0.4417    0.5621       369
 I-AntidilutiveSecuritiesExcludedFromComputationOfEarningsPerShareAmount     1.0000    1.0000    1.0000         1
                                           I-CommonStockSharesAuthorized     1.0000    1.0000    1.0000         1
                          I-SaleOfStockNumberOfSharesIssuedInTransaction     0.0000    0.0000    0.0000         2
                                I-StockIssuedDuringPeriodSharesNewIssues     0.0000    0.0000    0.0000         7
                                                                       O     0.9991    0.9978    0.9984     97395
                                                                accuracy       -          -      0.9938    100717
                                                               macro-avg     0.7519    0.7719    0.7506    100717
                                                            weighted-avg     0.9950    0.9938    0.9940    100717


```