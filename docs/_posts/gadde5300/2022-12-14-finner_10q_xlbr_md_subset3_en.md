---
layout: model
title: Finance NER (10Q, md, 12 entities)
author: John Snow Labs
name: finner_10q_xlbr_md_subset3
date: 2022-12-14
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

`ContractWithCustomerLiability`, `DebtInstrumentFairValue`, `DebtInstrumentMaturityDate`, `DebtInstrumentInterestRateStatedPercentage`, `DebtInstrumentRedemptionPricePercentage`, `DebtInstrumentInterestRateEffectivePercentage`, `DebtInstrumentBasisSpreadOnVariableRate1`, `CumulativeEffectOfNewAccountingPrincipleInPeriodOfAdoption`, `DebtInstrumentConvertibleConversionPrice1`, `ContractWithCustomerLiabilityRevenueRecognized`, `DebtInstrumentCarryingAmount`, `DebtInstrumentFaceAmount`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/finance/models/finner_10q_xlbr_md_subset3_en_1.0.0_3.0_1671036722936.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

nerTagger = finance.NerModel.pretrained('finner_10q_xlbr_md_subset3', 'en', 'finance/models')\
   .setInputCols(["sentence", "token", "embeddings"])\
   .setOutputCol("ner")
              
pipeline = nlp.Pipeline(stages=[documentAssembler,
                            sentence,
                            tokenizer,
                            embeddings,
                            nerTagger
                                ])
text = "Common Stock The authorized capital of the Company is 200,000,000 common shares , par value $ 0.001 , of which 12,481,724 are issued or outstanding ."

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

+----------+--------------------------+----------+
|token     |ner_label                 |confidence|
+----------+--------------------------+----------+
|Notes     |O                         |1.0       |
|and       |O                         |1.0       |
|Other     |O                         |1.0       |
|Debts     |O                         |1.0       |
|Payable   |O                         |1.0       |
|In        |O                         |1.0       |
|November  |O                         |1.0       |
|2013      |O                         |0.9999    |
|,         |O                         |1.0       |
|the       |O                         |1.0       |
|Rialto    |O                         |1.0       |
|segment   |O                         |1.0       |
|originally|O                         |1.0       |
|issued    |O                         |1.0       |
|$         |O                         |0.9999    |
|250       |B-DebtInstrumentFaceAmount|0.5981    |
|million   |O                         |1.0       |
|aggregate |O                         |1.0       |
|principal |O                         |1.0       |
|amount    |O                         |1.0       |
+----------+--------------------------+----------+

```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|finner_10q_xlbr_md_subset3|
|Compatibility:|Finance NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|en|
|Size:|16.5 MB|

## References

An in-house modified version of https://huggingface.co/datasets/nlpaueb/finer-139, re-splited and filtered to focus on sentences with bigger density of tags.

## Benchmarking

```bash

label                                                           precision    recall  f1-score   support
B-ContractWithCustomerLiability                                    0.9847    0.8344    0.9033       308
B-ContractWithCustomerLiabilityRevenueRecognized                   0.9657    0.9888    0.9771       627
B-CumulativeEffectOfNewAccountingPrincipleInPeriodOfAdoption       0.9817    0.9267    0.9534       232
B-DebtInstrumentBasisSpreadOnVariableRate1                         0.9332    0.9828    0.9573      2145
B-DebtInstrumentCarryingAmount                                     0.7082    0.4751    0.5687       562
B-DebtInstrumentConvertibleConversionPrice1                        0.9770    0.9290    0.9524       183
B-DebtInstrumentFaceAmount                                         0.8648    0.7501    0.8034      1901
B-DebtInstrumentFairValue                                          0.9728    0.7150    0.8242       200
B-DebtInstrumentInterestRateEffectivePercentage                    0.8225    0.8024    0.8123       410
B-DebtInstrumentInterestRateStatedPercentage                       0.9410    0.9454    0.9432      2731
B-DebtInstrumentMaturityDate                                       0.9143    0.4812    0.6305       266
B-DebtInstrumentRedemptionPricePercentage                          0.9518    0.9518    0.9518       436
I-DebtInstrumentMaturityDate                                       0.9143    0.4812    0.6305       798
O                                                                  0.9947    0.9984    0.9966    246872
accuracy                                                               -          -    0.9917    257671
macro-avg                                                          0.9233    0.8045    0.8504    257671
weighted-avg                                                       0.9912    0.9917    0.9911    257671
```