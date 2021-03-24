---
layout: model
title: Translate English to Pijin Pipeline
author: John Snow Labs
name: translate_en_pis
date: 2021-01-03
task: [Translation, Pipeline Public]
language: pis
edition: Spark NLP 2.7.0
tags: [open_source, seq2seq, translation, pipeline, en, pis, xx]
supported: false
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Marian is an efficient, free Neural Machine Translation framework written in pure C++ with minimal dependencies. It is mainly being developed by the Microsoft Translator team. Many academic (most notably the University of Edinburgh and in the past the Adam Mickiewicz University in Poznań) and commercial contributors help with its development.

It is currently the engine behind the Microsoft Translator Neural Machine Translation services and being deployed by many companies, organizations and research projects (see below for an incomplete list).

Note that this is a very computationally expensive module especially on larger sequence. The use of an accelerator such as GPU is recommended.

- source languages: `en`

- target languages: `pis`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/translate_en_pis_xx_2.7.0_2.4_1609698832184.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline 
pipeline = PretrainedPipeline("translate_en_pis", lang = "xx") 
pipeline.annotate("Your sentence to translate!")
```
```scala

import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val pipeline = new PretrainedPipeline("translate_en_pis", lang = "xx")

pipeline.annotate("Your sentence to translate!")
```

{:.nlu-block}
```python
import nlu

text = ["text to translate"]
translate_df = nlu.load('xx.en.translate_to.pis').predict(text, output_level='sentence')
translate_df
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|translate_en_pis|
|Type:|pipeline|
|Compatibility:|Spark NLP 2.7.0+|
|Edition:|Official|
|Language:|xx|

## Data Source

[https://huggingface.co/Helsinki-NLP/opus-mt-en-pis](https://huggingface.co/Helsinki-NLP/opus-mt-en-pis)