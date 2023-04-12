---
layout: model
title: Translate English to Congo Swahili Pipeline
author: John Snow Labs
name: translate_en_swc
date: 2021-01-03
task: [Translation, Pipeline Public]
language: xx
edition: Spark NLP 2.7.0
spark_version: 2.4
tags: [open_source, seq2seq, translation, pipeline, en, swc, xx]
supported: true
annotator: PipelineModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Marian is an efficient, free Neural Machine Translation framework written in pure C++ with minimal dependencies. It is mainly being developed by the Microsoft Translator team. Many academic (most notably the University of Edinburgh and in the past the Adam Mickiewicz University in Poznań) and commercial contributors help with its development.

It is currently the engine behind the Microsoft Translator Neural Machine Translation services and being deployed by many companies, organizations and research projects (see below for an incomplete list).

Note that this is a very computationally expensive module especially on larger sequence. The use of an accelerator such as GPU is recommended.

- source languages: `en`

- target languages: `swc`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/translate_en_swc_xx_2.7.0_2.4_1609687714800.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/translate_en_swc_xx_2.7.0_2.4_1609687714800.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline 
pipeline = PretrainedPipeline("translate_en_swc", lang = "xx") 
pipeline.annotate("Your sentence to translate!")
```
```scala

import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val pipeline = new PretrainedPipeline("translate_en_swc", lang = "xx")

pipeline.annotate("Your sentence to translate!")
```

{:.nlu-block}
```python
import nlu

text = ["text to translate"]
translate_df = nlu.load('xx.en.translate_to.swc').predict(text, output_level='sentence')
translate_df
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|translate_en_swc|
|Type:|pipeline|
|Compatibility:|Spark NLP 2.7.0+|
|Edition:|Official|
|Language:|xx|

## Data Source

[https://github.com/Helsinki-NLP/OPUS-MT-train/tree/master/models](https://github.com/Helsinki-NLP/OPUS-MT-train/tree/master/models)