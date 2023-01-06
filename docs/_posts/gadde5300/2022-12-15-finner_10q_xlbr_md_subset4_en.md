---
layout: model
title: Finance NER (10Q, md, 12 entities)
author: John Snow Labs
name: finner_10q_xlbr_md_subset4
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

`DeferredFinanceCostsNet`, `DebtInstrumentUnamortizedDiscount`, `EffectiveIncomeTaxRateContinuingOperations`, `DefinedContributionPlanCostRecognized`, `DefinedBenefitPlanContributionsByEmployer`, `DebtInstrumentTerm`, `Depreciation`, `DerivativeNotionalAmount`, `DisposalGroupIncludingDiscontinuedOperationConsideration`, `DeferredFinanceCostsGross`, `DerivativeFixedInterestRate`, `DebtWeightedAverageInterestRate`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/finance/models/finner_10q_xlbr_md_subset4_en_1.0.0_3.0_1671079076201.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

nerTagger = finance.NerModel.pretrained('finner_10q_xlbr_md_subset4', 'en', 'finance/models')\
   .setInputCols(["sentence", "token", "embeddings"])\
   .setOutputCol("ner")
              
pipeline = nlp.Pipeline(stages=[documentAssembler,
                            sentence,
                            tokenizer,
                            embeddings,
                            nerTagger
                                ])
text = "Depreciation expense for the six months ended May 31 , 2016 and May 31 , 2015 amounted to $ 38,919 and $ 104,790 , respectively ."

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

+------------+--------------+----------+
|token       |ner_label     |confidence|
+------------+--------------+----------+
|Depreciation|O             |1.0       |
|expense     |O             |1.0       |
|for         |O             |1.0       |
|the         |O             |1.0       |
|six         |O             |1.0       |
|months      |O             |1.0       |
|ended       |O             |1.0       |
|May         |O             |1.0       |
|31          |O             |1.0       |
|,           |O             |1.0       |
|2016        |O             |1.0       |
|and         |O             |1.0       |
|May         |O             |1.0       |
|31          |O             |1.0       |
|,           |O             |1.0       |
|2015        |O             |1.0       |
|amounted    |O             |1.0       |
|to          |O             |1.0       |
|$           |O             |1.0       |
|38,919      |B-Depreciation|0.9993    |
+------------+--------------+----------+

```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|finner_10q_xlbr_md_subset4|
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

label                                                       precision    recall  f1-score   support
B-DebtInstrumentTerm                                           0.9520    0.9754    0.9636       122
B-DebtInstrumentUnamortizedDiscount                            0.9381    0.9479    0.9430       192
B-DebtWeightedAverageInterestRate                              0.9541    0.9842    0.9689       190
B-DeferredFinanceCostsGross                                    0.6897    0.8000    0.7407       150
B-DeferredFinanceCostsNet                                      0.8252    0.8369    0.8310       282
B-DefinedBenefitPlanContributionsByEmployer                    0.9864    0.8286    0.9006       350
B-DefinedContributionPlanCostRecognized                        0.8298    0.9845    0.9006       322
B-Depreciation                                                 0.9645    1.0000    0.9819       598
B-DerivativeFixedInterestRate                                  0.9254    0.9841    0.9538       189
B-DerivativeNotionalAmount                                     0.9521    0.9776    0.9647       671
B-DisposalGroupIncludingDiscontinuedOperationConsideration     0.9750    0.9750    0.9750       200
B-EffectiveIncomeTaxRateContinuingOperations                   0.9958    1.0000    0.9979      1199
I-DebtInstrumentTerm                                           0.9583    0.9388    0.9485        49
O                                                              0.9996    0.9986    0.9991     95616
accuracy                                                            -         -    0.9968    100130
macro-avg                                                      0.9247    0.9451    0.9335    100130
weighted-avg                                                   0.9970    0.9968    0.9969    100130
```