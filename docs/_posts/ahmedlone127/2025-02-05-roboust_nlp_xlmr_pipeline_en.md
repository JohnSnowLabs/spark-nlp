---
layout: model
title: English roboust_nlp_xlmr_pipeline pipeline XlmRoBertaEmbeddings from Blue7Bird
author: John Snow Labs
name: roboust_nlp_xlmr_pipeline
date: 2025-02-05
tags: [en, open_source, pipeline, onnx]
task: Embeddings
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

Pretrained XlmRoBertaEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`roboust_nlp_xlmr_pipeline` is a English model originally trained by Blue7Bird.

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/roboust_nlp_xlmr_pipeline_en_5.5.1_3.0_1738793754742.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/roboust_nlp_xlmr_pipeline_en_5.5.1_3.0_1738793754742.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
pipeline = PretrainedPipeline("roboust_nlp_xlmr_pipeline", lang = "en")
annotations =  pipeline.transform(df)
```
```scala
val pipeline = new PretrainedPipeline("roboust_nlp_xlmr_pipeline", lang = "en")
val annotations = pipeline.transform(df)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|roboust_nlp_xlmr_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|655.2 MB|

## References

References

https://huggingface.co/Blue7Bird/Roboust_nlp_xlmr

## Included Models

- DocumentAssembler
- TokenizerModel
- XlmRoBertaForSequenceClassification