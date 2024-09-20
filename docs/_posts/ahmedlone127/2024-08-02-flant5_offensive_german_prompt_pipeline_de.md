---
layout: model
title: German flant5_offensive_german_prompt_pipeline pipeline T5Transformer from JenniferHJF
author: John Snow Labs
name: flant5_offensive_german_prompt_pipeline
date: 2024-08-02
tags: [de, open_source, pipeline, onnx]
task: [Question Answering, Summarization, Translation, Text Generation]
language: de
edition: Spark NLP 5.4.2
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained T5Transformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`flant5_offensive_german_prompt_pipeline` is a German model originally trained by JenniferHJF.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/flant5_offensive_german_prompt_pipeline_de_5.4.2_3.0_1722580118758.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/flant5_offensive_german_prompt_pipeline_de_5.4.2_3.0_1722580118758.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("flant5_offensive_german_prompt_pipeline", lang = "de")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("flant5_offensive_german_prompt_pipeline", lang = "de")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|flant5_offensive_german_prompt_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.4.2+|
|License:|Open Source|
|Edition:|Official|
|Language:|de|
|Size:|1.0 GB|

## References

https://huggingface.co/JenniferHJF/flant5_offensive_German_prompt

## Included Models

- DocumentAssembler
- T5Transformer