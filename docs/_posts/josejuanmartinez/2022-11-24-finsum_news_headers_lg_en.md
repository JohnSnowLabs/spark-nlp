---
layout: model
title: Financial News Summarization (Headers, Large)
author: John Snow Labs
name: finsum_news_headers_lg
date: 2022-11-24
tags: [financial, summarization, summary, en, licensed]
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

This model is a Financial news Summarizer, aimed to extract headers from financial news. This is a `lg` (large) version. Other versions may be found in Models Hub.

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/finance/models/finsum_news_headers_lg_en_1.0.0_3.0_1669287223603.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
document_assembler = nlp.DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("documents")

t5 = nlp.T5Transformer() \
    .pretrained("finsum_news_headers_lg" ,"en", "finance/models") \
    .setTask("summarization") \
    .setInputCols(["documents"]) \
    .setMaxOutputLength(512) \
    .setOutputCol("summaries")
    
data_df = spark.createDataFrame([["FTX is expected to make its debut appearance Tuesday in Delaware bankruptcy court, where its new management is expected to recount events leading up to the cryptocurrency platform’s sudden collapse and explain the steps it has since taken to secure customer funds and other assets."]]).toDF("text")

pipeline = Pipeline().setStages([document_assembler, t5])
results = pipeline.fit(data_df).transform(data_df)
results.select("summaries.result").show(truncate=False)
```

</div>

## Results

```bash
FTX to Make Debut in Delaware Bankruptcy Court Tuesday.
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|finsum_news_headers_lg|
|Compatibility:|Finance NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[documents]|
|Output Labels:|[summaries]|
|Language:|en|
|Size:|925.6 MB|

## References

In-house JSL financial summarized news.
