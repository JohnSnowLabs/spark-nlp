---
layout: model
title: Financial News Summarization (X-Small)
author: John Snow Labs
name: finsum_news_xs
date: 2022-11-23
tags: [financial, summarization, en, licensed]
task: Summarization
language: en
edition: Finance NLP 1.0.0
spark_version: 3.0
supported: true
annotator: T5Transformer
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model is a Financial news Summarizer, finetuned with a extra-small financial dataset. (about 4K news).

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/finance/models/finsum_news_xs_en_1.0.0_3.0_1669213220483.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
document_assembler = nlp.DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("documents")

t5 = nlp.T5Transformer() \
    .pretrained("finsum_news_xs" ,"en", "finance/models") \
    .setTask("summarize:")\ # or 'summarization'
    .setMaxOutputLength(512)\
    .setInputCols(["documents"]) \
    .setOutputCol("summaries")

data_df = spark.createDataFrame([["Deere Grows Sales 37% as Shipments Rise. Farm equipment supplier forecasts higher sales in year ahead, lifted by price increases and infrastructure investments. Deere & Co. said its fiscal fourth-quarter sales surged 37% as supply constraints eased and the company shipped more of its farm and construction equipment. The Moline, Ill.-based company, the largest supplier of farm equipment in the U.S., said demand held up as it raised prices on farm equipment, and forecast sales gains in the year ahead. Chief Executive John May cited strong demand and increased investment in infrastructure projects as the Biden administration ramps up spending. Elevated crop prices have kept farmers interested in new machinery even as their own production expenses increase."]]).toDF("text")

pipeline = Pipeline().setStages([document_assembler, t5])
results = pipeline.fit(data_df).transform(data_df)
results.select("summaries.result").show(truncate=False)
```

</div>

## Results

```bash
Deere & Co. said its fiscal fourth-quarter sales surged 37% as supply constraints eased and the company shipped more farm and construction equipment. Deere & Co. said its fiscal fourth-quarter sales surged 37% as supply constraints eased and the company shipped more farm and construction equipment.
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|finsum_news_xs|
|Compatibility:|Finance NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[documents]|
|Output Labels:|[summaries]|
|Language:|en|
|Size:|923.2 MB|

## Benchmarking

```bash
John Snow Labs in-house summarized articles.
```
