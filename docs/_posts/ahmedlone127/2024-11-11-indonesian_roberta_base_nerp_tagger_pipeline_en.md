---
layout: model
title: English indonesian_roberta_base_nerp_tagger_pipeline pipeline RoBertaForTokenClassification from w11wo
author: John Snow Labs
name: indonesian_roberta_base_nerp_tagger_pipeline
date: 2024-11-11
tags: [en, open_source, pipeline, onnx]
task: Named Entity Recognition
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

Pretrained RoBertaForTokenClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`indonesian_roberta_base_nerp_tagger_pipeline` is a English model originally trained by w11wo.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/indonesian_roberta_base_nerp_tagger_pipeline_en_5.5.1_3.0_1731311391967.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/indonesian_roberta_base_nerp_tagger_pipeline_en_5.5.1_3.0_1731311391967.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("indonesian_roberta_base_nerp_tagger_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("indonesian_roberta_base_nerp_tagger_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|indonesian_roberta_base_nerp_tagger_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|465.5 MB|

## References

https://huggingface.co/w11wo/indonesian-roberta-base-nerp-tagger

## Included Models

- DocumentAssembler
- TokenizerModel
- RoBertaForTokenClassification