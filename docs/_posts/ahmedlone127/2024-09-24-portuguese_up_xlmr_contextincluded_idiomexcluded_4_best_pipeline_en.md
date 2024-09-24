---
layout: model
title: English portuguese_up_xlmr_contextincluded_idiomexcluded_4_best_pipeline pipeline XlmRoBertaForSequenceClassification from harish
author: John Snow Labs
name: portuguese_up_xlmr_contextincluded_idiomexcluded_4_best_pipeline
date: 2024-09-24
tags: [en, open_source, pipeline, onnx]
task: Text Classification
language: en
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained XlmRoBertaForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`portuguese_up_xlmr_contextincluded_idiomexcluded_4_best_pipeline` is a English model originally trained by harish.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/portuguese_up_xlmr_contextincluded_idiomexcluded_4_best_pipeline_en_5.5.0_3.0_1727153550979.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/portuguese_up_xlmr_contextincluded_idiomexcluded_4_best_pipeline_en_5.5.0_3.0_1727153550979.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("portuguese_up_xlmr_contextincluded_idiomexcluded_4_best_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("portuguese_up_xlmr_contextincluded_idiomexcluded_4_best_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|portuguese_up_xlmr_contextincluded_idiomexcluded_4_best_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|788.2 MB|

## References

https://huggingface.co/harish/PT-UP-xlmR-ContextIncluded_IdiomExcluded-4_BEST

## Included Models

- DocumentAssembler
- TokenizerModel
- XlmRoBertaForSequenceClassification