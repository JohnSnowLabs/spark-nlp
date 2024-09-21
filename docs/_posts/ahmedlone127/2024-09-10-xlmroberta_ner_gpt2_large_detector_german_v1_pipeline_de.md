---
layout: model
title: German xlmroberta_ner_gpt2_large_detector_german_v1_pipeline pipeline XlmRoBertaForTokenClassification from bettertextapp
author: John Snow Labs
name: xlmroberta_ner_gpt2_large_detector_german_v1_pipeline
date: 2024-09-10
tags: [de, open_source, pipeline, onnx]
task: Named Entity Recognition
language: de
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained XlmRoBertaForTokenClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`xlmroberta_ner_gpt2_large_detector_german_v1_pipeline` is a German model originally trained by bettertextapp.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/xlmroberta_ner_gpt2_large_detector_german_v1_pipeline_de_5.5.0_3.0_1726012170831.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/xlmroberta_ner_gpt2_large_detector_german_v1_pipeline_de_5.5.0_3.0_1726012170831.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("xlmroberta_ner_gpt2_large_detector_german_v1_pipeline", lang = "de")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("xlmroberta_ner_gpt2_large_detector_german_v1_pipeline", lang = "de")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|xlmroberta_ner_gpt2_large_detector_german_v1_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|de|
|Size:|810.6 MB|

## References

https://huggingface.co/bettertextapp/gpt2-large-detector-de-v1

## Included Models

- DocumentAssembler
- TokenizerModel
- XlmRoBertaForTokenClassification