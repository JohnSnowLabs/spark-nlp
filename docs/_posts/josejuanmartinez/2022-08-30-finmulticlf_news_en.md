---
layout: model
title: Financial News Multilabel Classifier
author: John Snow Labs
name: finmulticlf_news
date: 2022-08-30
tags: [en, finance, classification, news, licensed]
task: Text Classification
language: en
edition: Finance NLP 1.0.0
spark_version: 3.0
supported: true
annotator: MultiClassifierDLModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This is a Multilabel classification model trained on different news scrapped from the internet and in-house annotations and label grouping. As this model is Multilabel, you can get as an output of a financial new, an array of 0 (no classes detected), 1(one class) or N (n classes detected).

The available classes are:

- acq: Acquisition / Purchase operations
- finance: Generic financial news
- fuel: News about fuel and energy sources
- jobs: News about jobs, employment rates, etc.
- livestock: News about animales and livestock
- mineral: News about mineral as copper, gold, silver, coal, etc.
- plant: News about greens, plants, cereals, etc
- trade: Trading news

## Predicted Entities

`acq`, `finance`, `fuel`, `jobs`, `livestock`, `mineral`, `plant`, `trade`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/finance/CLASSIFICATION_MULTILABEL/){:.button.button-orange}
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/finance/models/finmulticlf_news_en_1.0.0_3.2_1661857631377.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/finance/models/finmulticlf_news_en_1.0.0_3.2_1661857631377.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
documentAssembler = nlp.DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document") \
    .setCleanupMode("shrink")

embeddings = nlp.UniversalSentenceEncoder.pretrained() \
    .setInputCols("document") \
    .setOutputCol("embeddings")

docClassifier = nlp.MultiClassifierDLModel.pretrained("finmulticlf_news", "en","finance/models")\
    .setInputCols("embeddings") \
    .setOutputCol("category")

pipeline = Pipeline() \
    .setStages(
      [
        documentAssembler,
        embeddings,
        docClassifier
      ]
    )

empty_data = spark.createDataFrame([[""]]).toDF("text")

pipelineModel = pipeline.fit(empty_data)

text = ["""
ECUADOR HAS TRADE SURPLUS IN FIRST FOUR MONTHS Ecuador posted a trade surplus of 10.6 mln dlrs in the first four months of 1987 compared with a surplus of 271.7 mln in the same period in 1986, the central bank of Ecuador said in its latest monthly report. Ecuador suspended sales of crude oil, its principal export product, in March after an earthquake destroyed part of its oil-producing infrastructure. Exports in the first four months of 1987 were around 639 mln dlrs and imports 628.3 mln, compared with 771 mln and 500 mln respectively in the same period last year. Exports of crude and products in the first four months were around 256.1 mln dlrs, compared with 403.3 mln in the same period in 1986. The central bank said that between January and May Ecuador sold 16.1 mln barrels of crude and 2.3 mln barrels of products, compared with 32 mln and 2.7 mln respectively in the same period last year. Ecuador's international reserves at the end of May were around 120.9 mln dlrs, compared with 118.6 mln at the end of April and 141.3 mln at the end of May 1986, the central bank said. gold reserves were 165.7 mln dlrs at the end of May compared with 124.3 mln at the end of April.
"""]

lmodel = LightPipeline(pipelineModel)

results = lmodel.annotate(text)

```

</div>

## Results

```bash
['finance', 'trade']
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|finmulticlf_news|
|Compatibility:|Finance NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[embeddings]|
|Output Labels:|[category]|
|Language:|en|
|Size:|12.9 MB|

## References

News scrapped from the Internet and manual in-house annotations

## Benchmarking

```bash
       label  precision    recall  f1-score   support
         acq       0.94      0.92      0.93       718
     finance       0.95      0.96      0.96      1499
        fuel       0.91      0.86      0.88       286
        jobs       0.86      0.57      0.69        21
   livestock       0.93      0.44      0.60        57
     mineral       0.87      0.62      0.72       121
       plant       0.89      0.88      0.89       301
       trade       0.79      0.72      0.75       113
   micro-avg       0.93      0.90      0.92      3116
   macro-avg       0.89      0.75      0.80      3116
weighted-avg       0.93      0.90      0.91      3116
 samples-avg       0.91      0.91      0.91      3116
```
