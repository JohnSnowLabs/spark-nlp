---
layout: model
title: Classify Earning Calls, Broker Reports and 10K
author: John Snow Labs
name: finclf_earning_broker_10k
date: 2022-11-24
tags: [10k, earning, calls, broker, reports, en, licensed]
task: Text Classification
language: en
edition: Finance NLP 1.0.0
spark_version: 3.0
supported: true
annotator: FinanceClassifierDLModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This is a Text Cassification model, which can help you identify if a model is an `Earning Call`, a `Broker Report`, a `10K filing` or something else.

## Predicted Entities

`earning_call`, `broker_report`, `10k`, `other`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/finance/FINCLF_EARNING_BROKER_10K/){:.button.button-orange}
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/finance/models/finclf_earning_broker_10k_en_1.0.0_3.0_1669296495349.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = DocumentAssembler() \
  .setInputCol("text") \
  .setOutputCol("document")

embeddings = BertSentenceEmbeddings.pretrained("sent_bert_base_cased", "en") \
  .setInputCols("document") \
  .setOutputCol("sentence_embeddings")

docClassifier = finance.ClassifierDLModel.pretrained("finclf_earning_broker_10k", "en", "finance/models")\
    .setInputCols(["sentence_embeddings"])\
    .setOutputCol("label") \

nlpPipeline = Pipeline(stages=[
    documentAssembler, 
    embeddings,
    docClassifier])

text = """Varun Beverages  
 
 
Investors are advised to refer through important disclosures made at the last page of the Research Report.  
Motilal Oswal research is available on www.motilaloswal.com/Institutional -Equities, Bloomberg, Thomson Reuters, Factset and S&P Capital.  Research Analyst: Sumant Kumar (Sumant.Kumar@MotilalOswal.com)         
Research Analyst: Meet  Jain (Meet.Jain@ Motilal Oswal.com)  / Omkar Shintre  (Omkar.Shintre @Motilal Oswal.com)"""

sdf = spark.createDataFrame([[text]]).toDF("text")
fit = nlpPipeline.fit(sdf)
res = fit.transform(sdf)
res = res.select('label.result')
```

</div>

## Results

```bash
[broker_report]
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|finclf_earning_broker_10k|
|Compatibility:|Finance NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence_embeddings]|
|Output Labels:|[label]|
|Language:|en|
|Size:|22.8 MB|

## References

- Scrapped broker reports, earning calls, and 10K filings from the internet
- Other financial documents

## Benchmarking

```bash
        label  precision    recall  f1-score   support
          10k       1.00      1.00      1.00        17
broker_report       1.00      1.00      1.00        18
 earning_call       1.00      1.00      1.00        19
        other       1.00      1.00      1.00        98
     accuracy          -         -      1.00       152
    macro-avg       1.00      1.00      1.00       152
 weighted-avg       1.00      1.00      1.00       152
```
