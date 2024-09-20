---
layout: model
title: Azerbaijani sentiment_analysis_azerbaijani_pipeline pipeline XlmRoBertaForSequenceClassification from LocalDoc
author: John Snow Labs
name: sentiment_analysis_azerbaijani_pipeline
date: 2024-09-07
tags: [az, open_source, pipeline, onnx]
task: Text Classification
language: az
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained XlmRoBertaForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`sentiment_analysis_azerbaijani_pipeline` is a Azerbaijani model originally trained by LocalDoc.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/sentiment_analysis_azerbaijani_pipeline_az_5.5.0_3.0_1725711670642.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/sentiment_analysis_azerbaijani_pipeline_az_5.5.0_3.0_1725711670642.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("sentiment_analysis_azerbaijani_pipeline", lang = "az")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("sentiment_analysis_azerbaijani_pipeline", lang = "az")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sentiment_analysis_azerbaijani_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|az|
|Size:|863.9 MB|

## References

https://huggingface.co/LocalDoc/sentiment_analysis_azerbaijani

## Included Models

- DocumentAssembler
- TokenizerModel
- XlmRoBertaForSequenceClassification