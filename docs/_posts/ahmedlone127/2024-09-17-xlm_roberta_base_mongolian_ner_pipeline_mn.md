---
layout: model
title: Mongolian xlm_roberta_base_mongolian_ner_pipeline pipeline XlmRoBertaForTokenClassification from nemuwn
author: John Snow Labs
name: xlm_roberta_base_mongolian_ner_pipeline
date: 2024-09-17
tags: [mn, open_source, pipeline, onnx]
task: Named Entity Recognition
language: mn
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained XlmRoBertaForTokenClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`xlm_roberta_base_mongolian_ner_pipeline` is a Mongolian model originally trained by nemuwn.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/xlm_roberta_base_mongolian_ner_pipeline_mn_5.5.0_3.0_1726576566495.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/xlm_roberta_base_mongolian_ner_pipeline_mn_5.5.0_3.0_1726576566495.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("xlm_roberta_base_mongolian_ner_pipeline", lang = "mn")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("xlm_roberta_base_mongolian_ner_pipeline", lang = "mn")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|xlm_roberta_base_mongolian_ner_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|mn|
|Size:|842.2 MB|

## References

https://huggingface.co/nemuwn/xlm-roberta-base-mongolian-ner

## Included Models

- DocumentAssembler
- TokenizerModel
- XlmRoBertaForTokenClassification