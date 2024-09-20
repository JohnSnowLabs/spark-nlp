---
layout: model
title: Thai xlm_roberta_thai_2_pipeline pipeline XlmRoBertaForQuestionAnswering from milohpeng
author: John Snow Labs
name: xlm_roberta_thai_2_pipeline
date: 2024-09-06
tags: [th, open_source, pipeline, onnx]
task: Question Answering
language: th
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained XlmRoBertaForQuestionAnswering, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`xlm_roberta_thai_2_pipeline` is a Thai model originally trained by milohpeng.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/xlm_roberta_thai_2_pipeline_th_5.5.0_3.0_1725631403958.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/xlm_roberta_thai_2_pipeline_th_5.5.0_3.0_1725631403958.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("xlm_roberta_thai_2_pipeline", lang = "th")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("xlm_roberta_thai_2_pipeline", lang = "th")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|xlm_roberta_thai_2_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|th|
|Size:|881.1 MB|

## References

https://huggingface.co/milohpeng/xlm-roberta-th-2

## Included Models

- MultiDocumentAssembler
- XlmRoBertaForQuestionAnswering