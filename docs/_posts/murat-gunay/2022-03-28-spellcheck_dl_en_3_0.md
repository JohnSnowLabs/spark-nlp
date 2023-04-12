---
layout: model
title: Context Spell Checker for English
author: John Snow Labs
name: spellcheck_dl
date: 2022-03-28
tags: [spellcheck, en, open_source]
task: Spell Check
language: en
nav_key: models
edition: Spark NLP 3.4.1
spark_version: 3.0
supported: true
annotator: ContextSpellCheckerModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Spell Checker is a sequence-to-sequence model that detects and corrects spelling errors in your input text. It’s based on Levenshtein Automaton for generating candidate corrections and a Neural Language Model for ranking corrections.

## Predicted Entities



{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/CONTEXTUAL_SPELL_CHECKER/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/CONTEXTUAL_SPELL_CHECKER.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/spellcheck_dl_en_3.4.1_3.0_1648457196011.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/spellcheck_dl_en_3.4.1_3.0_1648457196011.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



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
.pretrained("spellcheck_dl", "en")\
.setInputCols("token")\
.setOutputCol("checked")\

pipeline = Pipeline(stages = [documentAssembler, tokenizer, spellModel])

empty_df = spark.createDataFrame([[""]]).toDF("text")
lp = LightPipeline(pipeline.fit(empty_df))
text = ["During the summer we have the best ueather.", "I have a black ueather jacket, so nice."]
lp.annotate(text)
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
pretrained("spellcheck_dl", "en").
setInputCols("token").
setOutputCol("checked")

val pipeline =  new Pipeline().setStages(Array(assembler, tokenizer, spellChecker))
val empty_df = spark.createDataFrame([[""]]).toDF("text")
val lp = new LightPipeline(pipeline.fit(empty_df))
val text = Array("During the summer we have the best ueather.", "I have a black ueather jacket, so nice.")
lp.annotate(text)
```


{:.nlu-block}
```python
import nlu
nlu.load("spell").predict("""During the summer we have the best ueather.""")
```

</div>

## Results

```bash
[{'checked': ['During', 'the', 'summer', 'we', 'have', 'the', 'best', 'weather', '.'],
'document': ['During the summer we have the best ueather.'],
'token': ['During', 'the', 'summer', 'we', 'have', 'the', 'best', 'ueather', '.']},

{'checked': ['I', 'have', 'a', 'black', 'leather', 'jacket', ',', 'so', 'nice',  '.'],
'document': ['I have a black ueather jacket, so nice.'],
'token': ['I', 'have', 'a', 'black', 'ueather', 'jacket', ',', 'so', 'nice', '.']}]
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|spellcheck_dl|
|Compatibility:|Spark NLP 3.4.1+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[token]|
|Output Labels:|[corrected]|
|Language:|en|
|Size:|99.7 MB|

## References

Combination of custom data sets.
