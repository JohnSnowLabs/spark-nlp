---
layout: model
title: Spell Checker in English Text
author: John Snow Labs
name: check_spelling_dl
date: 2022-01-04
tags: [open_source, en]
task: Spell Check
language: en
edition: Spark NLP 3.3.4
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Spell Checker is a sequence-to-sequence pipeline that detects and corrects spelling errors in your input text. It's based on Levenshtein Automaton for generating candidate corrections and a Neural Language Model for ranking corrections. You can download the pretrained pipeline that comes ready to use.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/check_spelling_dl_en_3.3.4_3.0_1641304582335.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
pipeline_local = PretrainedPipeline("clean_slang")

testDoc = '''
yo, what is wrong with ya?
'''

result=pipeline_local.annotate(testDoc)
```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val pipeline = new PretrainedPipeline("check_spelling_dl", lang = "en")
val result = pipeline.fullAnnotate("During the summer we have the hottest ueather. I have a black ueather jacket, so nice.I intrduce you to my sister, she is called ueather.")
```
</div>

## Results

```bash
['hey', 'what', 'is', 'wrong', 'with', 'you']
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|check_spelling_dl|
|Type:|pipeline|
|Compatibility:|Spark NLP 3.3.4+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|118.1 MB|

## Included Models

- DocumentAssembler
- SentenceDetectorDLModel
- TokenizerModel
- ContextSpellCheckerModel