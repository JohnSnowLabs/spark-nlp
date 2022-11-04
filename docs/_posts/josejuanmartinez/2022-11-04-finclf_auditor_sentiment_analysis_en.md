---
layout: model
title: Sentiment Analysis on Auditors' Reports
author: John Snow Labs
name: finclf_auditor_sentiment_analysis
date: 2022-11-04
tags: [auditor, sentiment, analysis, en, licensed]
task: Sentiment Analysis
language: en
edition: Finance NLP 1.0.0
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This is a Sentiment Analysis model which retrieves 3 sentiments (`positive`, `negative` or `neutral`) from Auditors' comments.

## Predicted Entities

`positive`, `negative`, `neutral`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/finance/models/finclf_auditor_sentiment_analysis_en_1.0.0_3.0_1667605773882.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = nlp.DocumentAssembler() \
    .setInputCol("sentence") \
    .setOutputCol("document")

embeddings = nlp.BertSentenceEmbeddings.pretrained("sent_bert_base_cased", "en") \
    .setInputCols("document") \
    .setOutputCol("sentence_embeddings")

sentiment =  nlp.ClassifierDLModel.pretrained("finclf_auditor_sentiment_analysis", "en", "finance/models") \
    .setInputCols("sentence_embeddings") \
    .setOutputCol("category")

pipeline = Pipeline() \
    .setStages(
      [
        documentAssembler,
        embeddings,
        sentiment 
      ]
    )

pipelineModel = pipeline.fit(sdf_test)
res = pipelineModel.transform(sdf_test)
res.select('sentence', 'category.result').show(truncate=100)
```

</div>

## Results

```bash
+----------------------------------------------------------------------------------------------------+----------+
|                                                                                            sentence|    result|
+----------------------------------------------------------------------------------------------------+----------+
|TeliaSonera TLSN said the offer is in line with its strategy to increase its ownership in core bu...|[positive]|
|STORA ENSO , NORSKE SKOG , M-REAL , UPM-KYMMENE Credit Suisse First Boston ( CFSB ) raised the fa...|[negative]|
|Clothing retail chain Sepp+ñl+ñ 's sales increased by 8 % to EUR 155.2 mn , and operating profit ...|[positive]|
|Lifetree was founded in 2000 , and its revenues have risen on an average by 40 % with margins in ...|[positive]|
|Nordea Group 's operating profit increased in 2010 by 18 percent year-on-year to 3.64 billion eur...|[positive]|
|Operating profit for the nine-month period increased from EUR3 .1 m and net sales increased from ...|[positive]|
|The Lithuanian beer market made up 14.41 million liters in January , a rise of 0.8 percent from t...|[positive]|
|In January-September 2010 , Fiskars ' net profit went up by 14 % year-on-year to EUR 65.4 million...|[positive]|
|Net income from life insurance rose to EUR 16.5 mn from EUR 14.0 mn , and net income from non-lif...|[positive]|
|Nyrstar has also agreed to supply to Talvivaara up to 150,000 tonnes of sulphuric acid per annum ...|[positive]|
|                   The agreement strengthens our long-term partnership with Nokia Siemens Networks .|[positive]|
|KESKO CORPORATION STOCK EXCHANGE RELEASE 28.02.2008 AT 09.30 1 ( 1 ) Kesko Corporation and Aspo p...| [neutral]|
|The OMX Helsinki 25 index was up 0.92 pct at 2,518.67 and the Helsinki CAP portfolio index was 0....|[positive]|
|Tiimari operates 194 stores in six countries -- including its core Finnish market -- and generate...| [neutral]|
|Under this agreement Biohit becomes a focus supplier of pipettors and disposable pipettor tips to...|[positive]|
|        Adjusted for changes in the Group structure , the Division 's net sales increased by 1.7 % .|[positive]|
|Finnish Aktia Group 's operating profit rose to EUR 17.5 mn in the first quarter of 2010 from EUR...|[positive]|
|Finnish high technology provider Vaahto Group reports net sales of EUR 41.8 mn in the accounting ...|[positive]|
|Biohit already services many current Genesis customers and the customer base is expected to expan...|[positive]|
|                     Circulation revenue has increased by 5 % in Finland and 4 % in Sweden in 2008 .|[positive]|
+----------------------------------------------------------------------------------------------------+----------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|finclf_auditor_sentiment_analysis|
|Compatibility:|Finance NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence_embeddings]|
|Output Labels:|[category]|
|Language:|en|
|Size:|23.1 MB|

## References

Propietary auditors' reports

## Benchmarking

```bash
label  precision    recall  f1-score   support
    negative       0.66      0.78      0.72       124
     neutral       0.88      0.77      0.82       559
    positive       0.65      0.76      0.70       286
    accuracy        -             -      0.77       969
   macro-avg       0.73      0.77      0.74       969
weighted-avg       0.78      0.77      0.77       969
```