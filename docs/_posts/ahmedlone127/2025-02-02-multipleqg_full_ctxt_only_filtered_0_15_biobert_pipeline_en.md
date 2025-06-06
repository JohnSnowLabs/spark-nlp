---
layout: model
title: English multipleqg_full_ctxt_only_filtered_0_15_biobert_pipeline pipeline BertForQuestionAnswering from LeWince
author: John Snow Labs
name: multipleqg_full_ctxt_only_filtered_0_15_biobert_pipeline
date: 2025-02-02
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

Pretrained BertForQuestionAnswering, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`multipleqg_full_ctxt_only_filtered_0_15_biobert_pipeline` is a English model originally trained by LeWince.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/multipleqg_full_ctxt_only_filtered_0_15_biobert_pipeline_en_5.5.1_3.0_1738496075599.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/multipleqg_full_ctxt_only_filtered_0_15_biobert_pipeline_en_5.5.1_3.0_1738496075599.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("multipleqg_full_ctxt_only_filtered_0_15_biobert_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("multipleqg_full_ctxt_only_filtered_0_15_biobert_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|multipleqg_full_ctxt_only_filtered_0_15_biobert_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|403.1 MB|

## References

https://huggingface.co/LeWince/MultipleQG-Full_Ctxt_Only-filtered_0_15_BioBert

## Included Models

- MultiDocumentAssembler
- BertForQuestionAnswering