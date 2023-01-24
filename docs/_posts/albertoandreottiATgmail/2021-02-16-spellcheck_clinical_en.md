---
layout: model
title: Medical Spell Checker
author: John Snow Labs
name: spellcheck_clinical
date: 2021-02-16
task: Spell Check
language: en
edition: Spark NLP 2.7.2
spark_version: 2.4
tags: [spelling, spellchecker, clinical, orthographic, spell_checker, medical_spell_checker, spelling_corrector, en, licensed]
supported: true
annotator: SpellCheckModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Contextual Spell Checker is a sequence-to-sequence model that detects and corrects spelling errors in your input text. It's based on Levenshtein Automaton for generating candidate corrections and a Neural Language Model for ranking corrections.
This model has been trained in a dataset containing data from different sources; MTSamples, i2b2 clinical notes, and PubMed. You can download the model that comes fully pretrained and ready to use. However, you can still customize it further without the need for re-training a new model from scratch. This can be accomplished by providing custom definitions for the word classes the model has been trained on, namely Dates, Numbers, Ages, Units, and Medications.

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/CONTEXTUAL_SPELL_CHECKER/){:.button.button-orange}
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/spellcheck_clinical_en_2.7.2_2.4_1613505168792.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/spellcheck_clinical_en_2.7.2_2.4_1613505168792.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use

In order to use this model, you need to setup a pipeline and feed tokens.

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = DocumentAssembler()\
.setInputCol("text")\
.setOutputCol("document")

tokenizer = RecursiveTokenizer()\
.setInputCols(["document"])\
.setOutputCol("token")\
.setPrefixes(["\"", "(", "[", "\n"])\
.setSuffixes([".", ",", "?", ")","!", "'s"])

spellModel = ContextSpellCheckerModel\
.pretrained('spellcheck_clinical', 'en', 'clinical/models')\
.setInputCols("token")\
.setOutputCol("checked")

finisher = Finisher()\
.setInputCols("checked")

pipeline = Pipeline(
stages = [
documentAssembler,
tokenizer,
spellModel,
finisher
])

empty_ds = spark.createDataFrame([[""]]).toDF("text")
lp = LightPipeline(pipeline.fit(empty_ds))

example = ["Witth the hell of phisical terapy the patient was imbulated and on posoperative, the impatient tolerating a post curgical soft diet.",
"With paint wel controlled on orall pain medications, she was discharged too reihabilitation facilitay.",
"She is to also call the ofice if she has any ever greater than 101, or leeding form the surgical wounds.",
"Abdomen is sort, nontender, and nonintended.",
"Patient not showing pain or any wealth problems.",
"No cute distress"

]
lp.annotate(example)

```



{:.nlu-block}
```python
import nlu
nlu.load("en.spell.clinical").predict(""")

pipeline = Pipeline(
stages = [
documentAssembler,
tokenizer,
spellModel,
finisher
])

empty_ds = spark.createDataFrame([[""")
```

</div>

## Results

```bash

[{'checked': ['With',
'the',
'help',
'of',
'physical',
'therapy',
'the',
'patient',
'was',
'ambulated',
'and',
'on',
'postoperative',
',',
'the',
'patient',
'tolerating',
'a',
'post',
'surgical',
'soft',
'diet',
'.']},
{'checked': ['With',
'pain',
'well',
'controlled',
'on',
'oral',
'pain',
'medications',
',',
'she',
'was',
'discharged',
'to',
'rehabilitation',
'facility',
'.']},
{'checked': ['She',
'is',
'to',
'also',
'call',
'the',
'office',
'if',
'she',
'has',
'any',
'fever',
'greater',
'than',
'101',
',',
'or',
'bleeding',
'from',
'the',
'surgical',
'wounds',
'.']},
{'checked': ['Abdomen',
'is',
'soft',
',',
'nontender',
',',
'and',
'nondistended',
'.']},
{'checked': ['Patient',
'not',
'showing',
'pain',
'or',
'any',
'health',
'problems',
'.']},
{'checked': ['No', 'acute', 'distress']}]
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|spellcheck_clinical|
|Compatibility:|Spark NLP 2.7.2+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[token]|
|Language:|en|

## Data Source

MTSamples, i2b2 clinical notes, and PubMed.