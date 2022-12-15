---
layout: model
title: Liability and Contra-Liability NER
author: John Snow Labs
name: finner_contraliability
date: 2022-12-15
tags: [en, finance, contra, liability, licensed, ner]
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

This is a financial model to detect LIABILITY and CONTRA_LIABILITY mentions in texts.  

- CONTRA_LIABILITY: Negative liability account that offsets the liability account (e.g. paying a debt)
- LIABILITY:  Current or Long-Term Liability (not from stockholders)

Please note this model requires some tokenization configuration to extract the currency (see python snippet below).

## Predicted Entities

`LIABILITY`, `CONTRA_LIABILITY`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/finance/models/finner_contraliability_en_1.0.0_3.0_1671136444267.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
document_assembler = nlp.DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

sentence_detector = nlp.SentenceDetectorDLModel.pretrained("sentence_detector_dl","xx")\
    .setInputCols(["document"])\
    .setOutputCol("sentence")

tokenizer = nlp.Tokenizer()\
    .setInputCols(["sentence"])\
    .setOutputCol("token")\
    .setContextChars(['.', ',', ';', ':', '!', '?', '*', '-', '(', ')', '”', '’', '$','€'])

embeddings = nlp.BertEmbeddings.pretrained("bert_embeddings_sec_bert_base", "en") \
  .setInputCols("sentence", "token") \
  .setOutputCol("embeddings")\
  .setMaxSentenceLength(512)

ner_model = finance.NerModel.pretrained("finner_contraliability", "en", "finance/models")\
    .setInputCols(["sentence", "token", "embeddings"])\
    .setOutputCol("ner")

ner_converter = nlp.NerConverter()\
    .setInputCols(["sentence", "token", "ner"])\
    .setOutputCol("ner_chunk")

pipeline = nlp.Pipeline(stages=[
    document_assembler,
    sentence_detector,
    tokenizer,
    embeddings,
    ner_model,
    ner_converter   
    ])

data = spark.createDataFrame([["""Reducing total debt continues to be a top priority , and we remain on track with our target of reducing overall debt levels by $ 15 billion by the end of 2025 ."""]]).toDF("text")

model = pipeline.fit(data)

result = model.transform(data)

result.select(F.explode(F.arrays_zip('ner_chunk.result', 'ner_chunk.metadata')).alias("cols")) \
               .select(F.expr("cols['0']").alias("text"),
                       F.expr("cols['1']['entity']").alias("label")).show(200, truncate = False)

```

</div>

## Results

```bash
+---------+------------------+----------+
|    token|         ner_label|confidence|
+---------+------------------+----------+
| Reducing|                 O|    0.9997|
|    total|       B-LIABILITY|    0.7884|
|     debt|       I-LIABILITY|    0.8479|
|continues|                 O|       1.0|
|       to|                 O|       1.0|
|       be|                 O|       1.0|
|        a|                 O|       1.0|
|      top|                 O|       1.0|
| priority|                 O|       1.0|
|        ,|                 O|       1.0|
|      and|                 O|       1.0|
|       we|                 O|       1.0|
|   remain|                 O|       1.0|
|       on|                 O|       1.0|
|    track|                 O|       1.0|
|     with|                 O|       1.0|
|      our|                 O|       1.0|
|   target|                 O|       1.0|
|       of|                 O|       1.0|
| reducing|                 O|    0.9993|
|  overall|                 O|    0.9969|
|     debt|B-CONTRA_LIABILITY|    0.5686|
|   levels|I-CONTRA_LIABILITY|    0.6611|
|       by|                 O|    0.9996|
|        $|                 O|       1.0|
|       15|                 O|       1.0|
|  billion|                 O|       1.0|
|       by|                 O|       1.0|
|      the|                 O|       1.0|
|      end|                 O|       1.0|
|       of|                 O|       1.0|
|     2025|                 O|       1.0|
|        .|                 O|       1.0|
+---------+------------------+----------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|finner_contraliability|
|Compatibility:|Finance NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|en|
|Size:|16.2 MB|

## References

In-house annotations on Earning Calls and 10-K Filings combined.

## Benchmarking

```bash
| label              | precision | recall | f1-score | support |
|--------------------|-----------|--------|----------|---------|
| B-CONTRA_LIABILITY | 0.7027    | 0.8125 | 0.7536   | 32      |
| B-LIABILITY        | 0.8182    | 0.6000 | 0.6923   | 15      |
| I-CONTRA_LIABILITY | 0.7143    | 0.6818 | 0.6977   | 22      |
| I-LIABILITY        | 0.5000    | 0.6667 | 0.5714   | 12      |
|                    |           |        |          |         |
| accuracy           |           |        | 0.9683   | 1261    |
| macro avg          | 0.7448    | 0.7493 | 0.7405   | 1261    |
| weighted avg       | 0.9702    | 0.9683 | 0.9688   | 1261    |
```