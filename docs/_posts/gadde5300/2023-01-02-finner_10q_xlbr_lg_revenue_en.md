---
layout: model
title: Finance Revenue NER (10Q, lg)
author: John Snow Labs
name: finner_10q_xlbr_lg_revenue
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

This model is an NER model containing 7 numeric financial Revenue entities from different 10Q reports. The tokens being annotated are the amounts, not any other surrounding word, but the context will determine what kind of amount is from the list of the 7 available.

This is a large (`lg`) model, trained with 200K sentences.

## Predicted Entities

`NumberOfReportableSegments`, `RevenueFromContractWithCustomerExcludingAssessedTax`, `ContractWithCustomerLiabilityRevenueRecognized`, `CumulativeEffectOfNewAccountingPrincipleInPeriodOfAdoption`, `NumberOfOperatingSegments`, `RevenueFromRelatedParties`, `RevenueFromContractWithCustomerIncludingAssessedTax`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/finance/models/finner_10q_xlbr_lg_revenue_en_1.0.0_3.0_1672653078889.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

nerTagger = finance.NerModel.pretrained('finner_10q_xlbr_lg_revenue', 'en', 'finance/models')\
   .setInputCols(["sentence", "token", "embeddings"])\
   .setOutputCol("ner")
              
pipeline = nlp.Pipeline(stages=[documentAssembler,
                            sentence,
                            tokenizer,
                            embeddings,
                            nerTagger
                                ])
text = "In addition , during the first quarter of 2016 , the Company made the decision to divide the Southeast Florida operating division into two operating segments to maximize operational efficiencies given the continued growth of the division ."

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

+------------+---------------------------+----------+
|token       |ner_label                  |confidence|
+------------+---------------------------+----------+
|In          |O                          |1.0       |
|addition    |O                          |1.0       |
|,           |O                          |1.0       |
|during      |O                          |1.0       |
|the         |O                          |1.0       |
|first       |O                          |1.0       |
|quarter     |O                          |1.0       |
|of          |O                          |1.0       |
|2016        |O                          |1.0       |
|,           |O                          |1.0       |
|the         |O                          |1.0       |
|Company     |O                          |1.0       |
|made        |O                          |1.0       |
|the         |O                          |1.0       |
|decision    |O                          |1.0       |
|to          |O                          |1.0       |
|divide      |O                          |1.0       |
|the         |O                          |1.0       |
|Southeast   |O                          |1.0       |
|Florida     |O                          |1.0       |
|operating   |O                          |1.0       |
|division    |O                          |1.0       |
|into        |O                          |1.0       |
|two         |B-NumberOfOperatingSegments|0.9831    |
|operating   |O                          |1.0       |
|segments    |O                          |1.0       |
|to          |O                          |1.0       |
|maximize    |O                          |1.0       |
|operational |O                          |1.0       |
|efficiencies|O                          |1.0       |
|given       |O                          |1.0       |
|the         |O                          |1.0       |
|continued   |O                          |1.0       |
|growth      |O                          |1.0       |
|of          |O                          |1.0       |
|the         |O                          |1.0       |
|division    |O                          |1.0       |
|.           |O                          |1.0       |
+------------+---------------------------+----------+

```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|finner_10q_xlbr_lg_revenue|
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

label                                                           precision    recall  f1-score   support
            B-ContractWithCustomerLiabilityRevenueRecognized     0.9516    0.9502    0.9509       642
B-CumulativeEffectOfNewAccountingPrincipleInPeriodOfAdoption     0.8588    0.9648    0.9087       227
                                 B-NumberOfOperatingSegments     0.8159    0.8654    0.8399       379
                                B-NumberOfReportableSegments     0.9186    0.8899    0.9040       672
       B-RevenueFromContractWithCustomerExcludingAssessedTax     0.7234    0.9223    0.8108       618
       B-RevenueFromContractWithCustomerIncludingAssessedTax     0.9130    0.2500    0.3925       168
                                 B-RevenueFromRelatedParties     0.9161    0.9192    0.9176       594
                                                           O     0.9998    0.9988    0.9993     80111
                                                    accuracy       -           -     0.9942     83411
                                                   macro-avg     0.8872    0.8451    0.8405     83411
                                                weighted-avg     0.9947    0.9942    0.9940     83411

```
