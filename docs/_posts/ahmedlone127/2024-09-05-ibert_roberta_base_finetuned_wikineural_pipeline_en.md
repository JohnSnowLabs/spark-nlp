---
layout: model
title: English ibert_roberta_base_finetuned_wikineural_pipeline pipeline RoBertaForTokenClassification from DunnBC22
author: John Snow Labs
name: ibert_roberta_base_finetuned_wikineural_pipeline
date: 2024-09-05
tags: [en, open_source, pipeline, onnx]
task: Named Entity Recognition
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

Pretrained RoBertaForTokenClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`ibert_roberta_base_finetuned_wikineural_pipeline` is a English model originally trained by DunnBC22.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/ibert_roberta_base_finetuned_wikineural_pipeline_en_5.5.0_3.0_1725512336086.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/ibert_roberta_base_finetuned_wikineural_pipeline_en_5.5.0_3.0_1725512336086.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("ibert_roberta_base_finetuned_wikineural_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("ibert_roberta_base_finetuned_wikineural_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ibert_roberta_base_finetuned_wikineural_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|462.0 MB|

## References

https://huggingface.co/DunnBC22/ibert-roberta-base-finetuned-WikiNeural

## Included Models

- DocumentAssembler
- TokenizerModel
- RoBertaForTokenClassification