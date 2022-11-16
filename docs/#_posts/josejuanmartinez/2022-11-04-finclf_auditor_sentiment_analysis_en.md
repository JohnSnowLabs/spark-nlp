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
annotator: FinanceClassifierDLModel
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
|In our opinion, the consolidated financial statements referred to above present fairly..............|[positive]|
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
    accuracy        -          -       0.77       969
   macro-avg       0.73      0.77      0.74       969
weighted-avg       0.78      0.77      0.77       969
```
