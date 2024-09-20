---
layout: model
title: English text_summarization_thirdeyedata_pipeline pipeline T5Transformer from ThirdEyeData
author: John Snow Labs
name: text_summarization_thirdeyedata_pipeline
date: 2024-08-07
tags: [en, open_source, pipeline, onnx]
task: [Question Answering, Summarization, Translation, Text Generation]
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

Pretrained T5Transformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`text_summarization_thirdeyedata_pipeline` is a English model originally trained by ThirdEyeData.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/text_summarization_thirdeyedata_pipeline_en_5.4.2_3.0_1723072541444.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/text_summarization_thirdeyedata_pipeline_en_5.4.2_3.0_1723072541444.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("text_summarization_thirdeyedata_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("text_summarization_thirdeyedata_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|text_summarization_thirdeyedata_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.4.2+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|302.1 MB|

## References

https://huggingface.co/ThirdEyeData/Text_Summarization

## Included Models

- DocumentAssembler
- T5Transformer