---
layout: model
title: Hindi xlsr_53_wav2vec_hindi_pipeline pipeline Wav2Vec2ForCTC from harshit345
author: John Snow Labs
name: xlsr_53_wav2vec_hindi_pipeline
date: 2025-04-04
tags: [hi, open_source, pipeline, onnx]
task: Automatic Speech Recognition
language: hi
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained Wav2Vec2ForCTC, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`xlsr_53_wav2vec_hindi_pipeline` is a Hindi model originally trained by harshit345.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/xlsr_53_wav2vec_hindi_pipeline_hi_5.5.1_3.0_1743808610391.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/xlsr_53_wav2vec_hindi_pipeline_hi_5.5.1_3.0_1743808610391.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("xlsr_53_wav2vec_hindi_pipeline", lang = "hi")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("xlsr_53_wav2vec_hindi_pipeline", lang = "hi")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|xlsr_53_wav2vec_hindi_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|hi|
|Size:|1.2 GB|

## References

https://huggingface.co/harshit345/xlsr-53-wav2vec-hi

## Included Models

- AudioAssembler
- Wav2Vec2ForCTC