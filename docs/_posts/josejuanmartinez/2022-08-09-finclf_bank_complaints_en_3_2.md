---
layout: model
title: Bank Complaints Classification
author: John Snow Labs
name: finclf_bank_complaints
date: 2022-08-09
tags: [en, finance, bank, classification, licensed]
task: Text Classification
language: en
edition: Finance NLP 1.0.0
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model classifies Bank-related texts into different 7 different categories, and can be used to automatically process incoming emails to customer support channels and forward them to the proper recipients.

## Predicted Entities

`Accounts`, `Credit Cards`, `Credit Reporting`, `Debt Collection`, `Loans`, `Money Transfer and Currency`, `Mortgage`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/finance/models/finclf_bank_complaints_en_1.0.0_3.2_1660035048303.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/finance/models/finclf_bank_complaints_en_1.0.0_3.2_1660035048303.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
document_assembler = nlp.DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")

embeddings = nlp.UniversalSentenceEncoder.pretrained() \
      .setInputCols("document") \
      .setOutputCol("sentence_embeddings")

classsifier_dl = nlp.ClassifierDLModel.pretrained("finclf_bank_complaints", "en", "finance/models")\
      .setInputCols(["sentence_embeddings"])\
      .setOutputCol("label")\

clf_pipeline = Pipeline(
    stages = [
        document_assembler,
        embeddings,
        classsifier_dl
    ])
    
light_pipeline = LightPipeline(clf_pipeline.fit(spark.createDataFrame([['']]).toDF("text")))

result = light_pipeline.annotate("""Over the course of 30 days I have filed a dispute in regards to inaccurate and false information on my credit report. Ive attached a copy of my dispute mailed in certified to Equifax and they are still reporting these incorrect items. According to the fair credit ACT, section 609 ( a ) ( 1 ) ( A ) they are required by Federal Law to only report Accurate information and the have not done so. They have not provided me with any proof i.e. and original consumer contract with my signature on it proving that this is my account.Further more, I would like to make a formal complaint that Ive tried calling Equifax Over 10 times this week and every single time Ive called Ive asked for a representative in the fraud dispute department wants transfer it over there when you speak to the representative and let them know that you are looking to dispute inquiries and accounts due to fraud they immediately transfer you to their survey line essentially ending the call. I believe Equifax is training their representatives to not help consumers over the phone and performing unethical practices. Once I finally got a hold of a representative she told me that she could not help because I did not send in my Social Security card which violates my consumer rights. So Im Making a formal CFPB complaint that you will correct Equifaxs actions. Below Ive written what is also included in the files uploaded, my disputes for inaccuracies on my credit report.""")

result['label']
```

</div>

## Results

```bash
['Credit Reporting']
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|finclf_bank_complaints|
|Compatibility:|Finance NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence_embeddings]|
|Output Labels:|[class]|
|Language:|en|
|Size:|22.4 MB|

## References

https://www.consumerfinance.gov/data-research/consumer-complaints/#download-the-data

## Benchmarking

```bash
                      label  precision    recall  f1-score   support
                   Accounts       0.77      0.73      0.75       490
               Credit_Cards       0.75      0.68      0.72       461
           Credit_Reporting       0.73      0.81      0.76       488
            Debt_Collection       0.72      0.72      0.72       459
                      Loans       0.78      0.78      0.78       472
Money_Transfer_and_Currency       0.82      0.84      0.83       482
                   Mortgage       0.87      0.87      0.87       488
                   accuracy         -         -       0.78      3340
                  macro-avg       0.78      0.78      0.78      3340
               weighted-avg       0.78      0.78      0.78      3340
```