---
layout: model
title: Extract Broker Recommendations from Broker Reports
author: John Snow Labs
name: finclf_bert_broker_recommendation
date: 2023-01-31
tags: [en, classification, licensed, bert, broker_reports, tensorflow]
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

This Text Classifier will identify whether a broker's report recommends a downgrade, maintain, upgrade, or other action.

## Predicted Entities

`Downgrade`, `Upgrade`, `Maintain`, `other`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/finance/models/finclf_bert_broker_recommendation_en_1.0.0_3.0_1675177545178.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/finance/models/finclf_bert_broker_recommendation_en_1.0.0_3.0_1675177545178.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
# Test classifier in Spark NLP pipeline
document_assembler = nlp.DocumentAssembler() \
    .setInputCol('text') \
    .setOutputCol('document')

tokenizer = nlp.Tokenizer() \
    .setInputCols(['document']) \
    .setOutputCol('token')

# Load newly trained classifier
sequenceClassifier_loaded = finance.BertForSequenceClassification.pretrained("finclf_bert_broker_recommendation", "en", "finance/models")\
  .setInputCols(["document",'token'])\
  .setOutputCol("class")

pipeline = nlp.Pipeline(stages=[
    document_assembler, 
    tokenizer,
    sequenceClassifier_loaded    
])

# Generating example
example = spark.createDataFrame([["""UPL 
   
 
Estimate change   
TP change   
Rating change   
 
Bloomberg  UPLL IN  
Equity Shares (m)  765 
M.Cap.(INRb)/(USDb)  538.2 / 6.5  
52-Week Range (INR)  848 / 608  
1, 6, 12 Rel. Per (%)  0/-20/-3  
12M Avg Val (INR M)  2009  
 
Financials & Valuation s (INR b)  
Y/E Mar  2022 2023E 2024E 
Sales  462.4  537.0  593.4  
EBITDA  101.7  121.5  135.3  
PAT 48.5  54.9  61.0  
EBITDA (%)           22.0           22.6            22.8  
EPS (INR)           63.5           71.7            79.7  
EPS Gr. (%)           39.9           13.0            11.1  
BV/Sh. (INR)  429 512 652 
Ratios        
Net D/E             1.0             0.8              0.5  
RoE (%)           24.5           23.1            20.7  
RoCE (%)           15.1           16.2            16.5  
Payout (%)           21.1           18.0            17.6  
Valuations        
P/E (x)           11.3           10.0              9.0  
EV/EBITDA (x)             7.6             6.3              5.2  
Div Yield (%)             1.4             1.7              2.0  
FCF Yield (%)             4.4             7.2            14.0  
 
Shareholding pattern (%)  
 Sep-22 Jun-22 Sep-21 
Promoter  29.0  29.0  28.0  
DII 17.2  16.5  18.0  
FII 42.8  36.4  35.1  
Others  11.1  18.1  19.0  
Note: FII includes depository receipts  
 
  CMP: INR 717                   TP: INR 780 (+9%)                       Neutral  
 
Higher working capital adversely impacts CFO  
Earnings better than expected    
 UPLL reported strong revenue growth of 18% YoY , driven primarily  by an 
increase in price realization ( up 21% YoY). However,  volume s declined (down 
7% YoY) in 2QFY23, led by rationalization of product mix toward high margin 
products. Except Europe (+1% YoY), all other key geographies registered a 
strong sales growth of over 20% YoY.  
 Gross debt increased to INR 326b in 2QFY23 from INR 301b in 1Q FY23 with 
net debt rising INR20b QoQ to INR 285b, due to an  increas e in working 
capital requirement . This increase in working capital also resulted in cash 
outflow from operation of INR45.94b in 1HFY23  v/s cash outflow INR24.15b 
in 1HFY22 .  
 We largely maintain our FY23E/FY24 E earnings . We reiterate our Neutral 
rating on the stock with a TP of INR 780 (premised on 1 0x FY24E P/E) ."""]]).toDF("text")

result = pipeline.fit(example).transform(example)

# Checking results
result.select("text", "class.result").show(truncate=False)
```

</div>

## Results

```bash
+---------+
|result   |
+---------+
|[Maintain]|
+---------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|finclf_bert_broker_recommendation|
|Compatibility:|Finance NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[class]|
|Language:|en|
|Size:|402.5 MB|
|Case sensitive:|true|
|Max sentence length:|128|

## References

An in-house annotated dataset of broker reports.

## Benchmarking

```bash
label          precision    recall  f1-score   support
   Downgrade       0.96      1.00      0.98        24
    Maintain       0.97      1.00      0.98        30
     Upgrade       1.00      0.94      0.97        16
       other       1.00      0.94      0.97        16
    accuracy        -         -        0.98        86
   macro-avg       0.98      0.97      0.97        86
weighted-avg       0.98      0.98      0.98        86
```
