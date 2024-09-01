---
layout: model
title: Multilingual ag_nli_dets_sentence_similarity_v2_pipeline pipeline RoBertaForSequenceClassification from abbasgolestani
author: John Snow Labs
name: ag_nli_dets_sentence_similarity_v2_pipeline
date: 2024-08-31
tags: [xx, open_source, pipeline, onnx]
task: Text Classification
language: xx
edition: Spark NLP 5.4.2
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained RoBertaForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`ag_nli_dets_sentence_similarity_v2_pipeline` is a Multilingual model originally trained by abbasgolestani.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/ag_nli_dets_sentence_similarity_v2_pipeline_xx_5.4.2_3.0_1725119734333.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/ag_nli_dets_sentence_similarity_v2_pipeline_xx_5.4.2_3.0_1725119734333.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("ag_nli_dets_sentence_similarity_v2_pipeline", lang = "xx")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("ag_nli_dets_sentence_similarity_v2_pipeline", lang = "xx")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ag_nli_dets_sentence_similarity_v2_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.4.2+|
|License:|Open Source|
|Edition:|Official|
|Language:|xx|
|Size:|1.3 GB|

## References

https://huggingface.co/abbasgolestani/ag-nli-DeTS-sentence-similarity-v2

## Included Models

- DocumentAssembler
- TokenizerModel
- RoBertaForSequenceClassification