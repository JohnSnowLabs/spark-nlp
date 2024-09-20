---
layout: model
title: English 20230327_baseline_xlmr_clickbait_spoiling_pipeline pipeline XlmRoBertaForQuestionAnswering from intanm
author: John Snow Labs
name: 20230327_baseline_xlmr_clickbait_spoiling_pipeline
date: 2024-09-01
tags: [en, open_source, pipeline, onnx]
task: Question Answering
language: en
edition: Spark NLP 5.4.2
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained XlmRoBertaForQuestionAnswering, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`20230327_baseline_xlmr_clickbait_spoiling_pipeline` is a English model originally trained by intanm.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/20230327_baseline_xlmr_clickbait_spoiling_pipeline_en_5.4.2_3.0_1725173862477.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/20230327_baseline_xlmr_clickbait_spoiling_pipeline_en_5.4.2_3.0_1725173862477.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("20230327_baseline_xlmr_clickbait_spoiling_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("20230327_baseline_xlmr_clickbait_spoiling_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|20230327_baseline_xlmr_clickbait_spoiling_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.4.2+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|875.9 MB|

## References

https://huggingface.co/intanm/20230327-baseline-xlmr-clickbait-spoiling

## Included Models

- MultiDocumentAssembler
- XlmRoBertaForQuestionAnswering