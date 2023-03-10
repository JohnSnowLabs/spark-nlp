---
layout: model
title: Finance-related Tweets Classifier
author: John Snow Labs
name: finclf_twitter_news
date: 2023-03-10
tags: [en, licensed, classifier, twitter, finance, tensorflow]
task: Text Classification
language: en
edition: Finance NLP 1.0.0
spark_version: 3.0
supported: true
engine: tensorflow
annotator: FinanceBertForSequenceClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This is a Multiclass classification model which classifies financial tweets with one of the following topics: `Company_or_Product_News`, `Stock_Movement`, `General_News_or_Opinion`, `Earnings`, `Macro`, `Fed_or_Central_Banks`, `Politics`, `Stock_Commentary`, `Financials`, `M&A_or_Investments`, `Legal_or_Regulation`, `Personnel_Change`, `Markets`, `Energy_or_Oil`, `Dividend`, `Analyst_Update`, `Treasuries_or_Corporate_Debt`, `Currencies`.

## Predicted Entities

`Company_or_Product_News`, `Stock_Movement`, `General_News_or_Opinion`, `Earnings`, `Macro`, `Fed_or_Central_Banks`, `Politics`, `Stock_Commentary`, `Financials`, `M&A_or_Investments`, `Legal_or_Regulation`, `Personnel_Change`, `Markets`, `Energy_or_Oil`, `Dividend`, `Analyst_Update`, `Treasuries_or_Corporate_Debt`, `Currencies`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/finance/models/finclf_twitter_news_en_1.0.0_3.0_1678444505428.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/finance/models/finclf_twitter_news_en_1.0.0_3.0_1678444505428.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = nlp.DocumentAssembler() \
    .setInputCols(["text"]) \
    .setOutputCols("document")

tokenizer = nlp.Tokenizer() \
    .setInputCols("document") \
    .setOutputCol("token")

seq_classifier = finance.BertForSequenceClassification.pretrained("finclf_twitter_news", "en", "finance/models") \
    .setInputCols(["document", "token"]) \
    .setOutputCol("class")
    
pipeline = nlp.Pipeline(stages=[documentAssembler, tokenizer, seq_classifier])

data = spark.createDataFrame([["Barclays believes earnings for these underperforming stocks may surprise Wall Street"]]).toDF("text")

result = pipeline.fit(data).transform(data)
```

</div>

## Results

```bash
+----------------+
|          result|
+----------------+
|[Analyst_Update]|
+----------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|finclf_twitter_news|
|Compatibility:|Finance NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[class]|
|Language:|en|
|Size:|408.7 MB|
|Case sensitive:|true|
|Max sentence length:|512|

## References

Train dataset available [here](https://huggingface.co/datasets/zeroshot/twitter-financial-news-topic)

## Benchmarking

```bash
label                         precision  recall  f1-score  support 
Analyst_Update                0.79       0.79    0.79      38      
Company_or_Product_News       0.71       0.78    0.74      112     
Currencies                    0.80       1.00    0.89      12      
Dividend                      1.00       0.94    0.97      31      
Earnings                      0.95       0.97    0.96      100     
Energy_or_Oil                 0.78       0.89    0.83      55      
Fed_or_Central_Banks          0.82       0.78    0.80      95      
Financials                    0.90       0.93    0.92      60      
General_News_or_Opinion       0.71       0.74    0.72      80      
Legal_or_Regulation           0.85       0.75    0.80      52      
M&A_or_Investments            0.85       0.90    0.87      49      
Macro                         0.81       0.70    0.75      84      
Markets                       0.91       0.84    0.87      49      
Personnel_Change              0.96       0.94    0.95      50      
Politics                      0.83       0.82    0.82      83      
Stock_Commentary              0.87       0.94    0.90      63      
Stock_Movement                0.94       0.90    0.92      89      
Treasuries_or_Corporate_Debt  0.80       0.73    0.76      33      
accuracy                      -          -       0.84      1135    
macro-avg                     0.85       0.85    0.85      1135    
weighted-avg                  0.84       0.84    0.84      1135    
```