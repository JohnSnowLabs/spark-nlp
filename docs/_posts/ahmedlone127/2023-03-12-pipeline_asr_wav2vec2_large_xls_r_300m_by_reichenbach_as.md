---
layout: model
title: Assamese asr_wav2vec2_large_xls_r_300m_by_reichenbach TFWav2Vec2ForCTC from reichenbach
author: John Snow Labs
name: pipeline_asr_wav2vec2_large_xls_r_300m_by_reichenbach
date: 2023-03-12
tags: [wav2vec2, as, audio, open_source, pipeline, asr]
task: Automatic Speech Recognition
language: as
edition: Spark NLP 4.4.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained Wav2vec2  pipeline, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`asr_wav2vec2_large_xls_r_300m_by_reichenbach` is a Assamese model originally trained by reichenbach.

NOTE: This pipeline only works on a CPU, if you need to use this pipeline on a GPU device please use pipeline_asr_wav2vec2_large_xls_r_300m_by_reichenbach_gpu

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/pipeline_asr_wav2vec2_large_xls_r_300m_by_reichenbach_as_4.4.0_3.0_1678625435378.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/pipeline_asr_wav2vec2_large_xls_r_300m_by_reichenbach_as_4.4.0_3.0_1678625435378.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

    pipeline = PretrainedPipeline('pipeline_asr_wav2vec2_large_xls_r_300m_by_reichenbach', lang = 'as')
    annotations =  pipeline.transform(audioDF)
    
```
```scala

    val pipeline = new PretrainedPipeline("pipeline_asr_wav2vec2_large_xls_r_300m_by_reichenbach", lang = "as")
    val annotations = pipeline.transform(audioDF)
    
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|pipeline_asr_wav2vec2_large_xls_r_300m_by_reichenbach|
|Type:|pipeline|
|Compatibility:|Spark NLP 4.4.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|as|
|Size:|1.2 GB|

## Included Models

- AudioAssembler
- Wav2Vec2ForCTC