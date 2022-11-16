---
layout: model
title: Context Spell Checker for the Italian Language
author: John Snow Labs
name: spellcheck_dl
date: 2021-03-08
tags: [it, open_source]
supported: true
task: Spell Check
language: it
edition: Spark NLP 2.7.4
spark_version: 2.4
annotator: ContextSpellCheckerModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This is an Italian Context Spell Checker trained on the Paisà corpus.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/spellcheck_dl_it_2.7.4_2.4_1615238955709.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use

The model works at the token level, so you must put it after tokenization.
The model can change the length of the tokes when correcting words, so keep this in mind when using it before other annotators that may work with absolute references to the original document like NerConverter.

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
assembler = DocumentAssembler()\
 .setInputCol("value")\
 .setOutputCol("document")

tokenizer = RecursiveTokenizer()\
 .setInputCols("document")\
 .setOutputCol("token")\
 .setPrefixes(["\"", """, "(", "[", "\n", ".", "l'", "dell'", "nell'", "sull'", "all'", "d'", "un'"])\
 .setSuffixes(["\"", """, ".", ",", "?", ")", "]", "!", ";", ":"])

spellChecker = ContextSpellCheckerModel("spellcheck_dl", "it").\
    setInputCols("token").\
    setOutputCol("corrected")
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|spellcheck_dl|
|Compatibility:|Spark NLP 2.7.4+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[token]|
|Output Labels:|[corrected]|
|Language:|it|

## Data Source

Paisà Italian Language Corpus.