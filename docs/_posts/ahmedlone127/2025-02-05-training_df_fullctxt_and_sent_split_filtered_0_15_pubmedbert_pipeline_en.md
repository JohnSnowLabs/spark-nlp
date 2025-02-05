---
layout: model
title: English training_df_fullctxt_and_sent_split_filtered_0_15_pubmedbert_pipeline pipeline BertForQuestionAnswering from LeWince
author: John Snow Labs
name: training_df_fullctxt_and_sent_split_filtered_0_15_pubmedbert_pipeline
date: 2025-02-05
tags: [en, open_source, pipeline, onnx]
task: Question Answering
language: en
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForQuestionAnswering, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`training_df_fullctxt_and_sent_split_filtered_0_15_pubmedbert_pipeline` is a English model originally trained by LeWince.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/training_df_fullctxt_and_sent_split_filtered_0_15_pubmedbert_pipeline_en_5.5.1_3.0_1738788718466.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/training_df_fullctxt_and_sent_split_filtered_0_15_pubmedbert_pipeline_en_5.5.1_3.0_1738788718466.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("training_df_fullctxt_and_sent_split_filtered_0_15_pubmedbert_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("training_df_fullctxt_and_sent_split_filtered_0_15_pubmedbert_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|training_df_fullctxt_and_sent_split_filtered_0_15_pubmedbert_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|53.7 MB|

## References

https://huggingface.co/LeWince/training_df_fullctxt_and_sent_split_filtered_0_15_PubMedBert

## Included Models

- MultiDocumentAssembler
- BertForQuestionAnswering