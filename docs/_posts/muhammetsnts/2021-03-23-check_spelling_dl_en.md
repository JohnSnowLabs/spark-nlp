---
layout: model
title: Spell Checker in English Text
author: John Snow Labs
name: check_spelling_dl
date: 2021-03-23
tags: [en, open_source]
task: Spell Check
language: en
edition: Spark NLP 2.7.5
spark_version: 2.4
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Spell Checker is a sequence-to-sequence pipeline that detects and corrects spelling errors in your input text. It's based on Levenshtein Automaton for generating candidate corrections and a Neural Language Model for ranking corrections. You can download the pretrained pipeline that comes ready to use.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/check_spelling_dl_en_2.7.5_2.4_1616486792751.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use

In order to use this pretrained pipeline, you need to just provide the text to be checked.

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline 

pipeline = PretrainedPipeline('check_spelling_dl', lang='en')
result = pipeline.fullAnnotate("Peter Parker is a very good persn.John and Peter are brothrs. However they don't spport each othr that much.")

```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val pipeline = new PretrainedPipeline("check_spelling_dl", lang = "en")
val result = pipeline.fullAnnotate("Peter Parker is a very good persn.John and Peter are brothrs. However they don't spport each othr that much.")
```
</div>

## Results

```bash
[('Peter', 'Peter'),
 ('Parker', 'Parker'),
 ('is', 'is'),
 ('a', 'a'),
 ('very', 'very'),
 ('good', 'good'),
 ('persn', 'person'),
 ('.', '.'),
 ('John', 'John'),
 ('and', 'and'),
 ('Peter', 'Peter'),
 ('are', 'are'),
 ('brothrs', 'brothers'),
 ('.', '.'),
 ('However', 'However'),
 ('they', 'they'),
 ("don't", "don't"),
 ('spport', 'support'),
 ('each', 'each'),
 ('othr', 'other'),
 ('that', 'that'),
 ('much', 'much'),
 ('.', '.')]
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|check_spelling_dl|
|Type:|pipeline|
|Compatibility:|Spark NLP 2.7.5+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|

## Included Models

`SentenceDetectorDLModel`
`ContextSpellCheckerModel`