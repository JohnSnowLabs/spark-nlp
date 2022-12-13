---
layout: model
title: Clinical Context Spell Checker
author: John Snow Labs
name: spellcheck_clinical
date: 2021-03-30
tags: [en, licensed]
supported: true
task: Spell Check
language: en
edition: Healthcare NLP 3.0.0
spark_version: 3.0
annotator: SpellCheckModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Spell Checker is a sequence-to-sequence model that detects and corrects spelling errors in your input text. It’s based on Levenshtein Automaton for generating candidate corrections and a Neural Language Model for ranking corrections.

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/CONTEXTUAL_SPELL_CHECKER/){:.button.button-orange}
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/spellcheck_clinical_en_3.0.0_3.0_1617128886628.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/spellcheck_clinical_en_3.0.0_3.0_1617128886628.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use

The model works at the token level, so you must put it after tokenization. The model can change the length of the tokens when correcting words, so keep this in mind when using it before other annotators that may work with absolute references to the original document like NerConverter.

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = DocumentAssembler()\
.setInputCol("text")\
.setOutputCol("document")

tokenizer = RecursiveTokenizer()\
.setInputCols(["document"])\
.setOutputCol("token")\
.setPrefixes(["\"", "“", "(", "[", "\n", "."]) \
.setSuffixes(["\"", "”", ".", ",", "?", ")", "]", "!", ";", ":", "'s", "’s"])

spellModel = ContextSpellCheckerModel\
.pretrained()\
.setInputCols("token")\
.setOutputCol("checked")\
```
```scala
val assembler = new DocumentAssembler()
.setInputCol("text")
.setOutputCol("document")

val tokenizer = new RecursiveTokenizer()
.setInputCols(Array("document"))
.setOutputCol("token")
.setPrefixes(Array("\"", "“", "(", "[", "\n", "."))
.setSuffixes(Array("\"", "”", ".", ",", "?", ")", "]", "!", ";", ":", "'s", "’s"))

val spellChecker = ContextSpellCheckerModel.
pretrained().
setInputCols("token").
setOutputCol("checked")
```


{:.nlu-block}
```python
import nlu
nlu.load("en.spell.clinical").predict("""]) \
.setSuffixes([""")
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|spellcheck_clinical|
|Compatibility:|Spark NLP 3.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[token]|
|Language:|en|

## Data Source

The dataset used contains data drawn from MT Samples clinical notes, i2b2 clinical notes, and PubMed abstracts.
