---
layout: model
title: Medical Spell Checker
author: John Snow Labs
name: spellcheck_clinical
date: 2022-04-11
tags: [spellcheck, medical, medical_spell_checker, spell_checker, spelling_corrector, en, licensed]
task: Spell Check
language: en
edition: Spark NLP for Healthcare 3.4.1
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Contextual Spell Checker is a sequence-to-sequence model that detects and corrects spelling errors in your medical input text. It’s based on Levenshtein Automation for generating candidate corrections and a Neural Language Model for ranking corrections. This model has been trained in a dataset containing data from different sources; MTSamples, i2b2 clinical notes, and several specific medical corpora. You can download the model that comes fully pretrained and ready to use. However, you can still customize it further without the need for re-training a new model from scratch. This can be accomplished by providing custom definitions for the word classes the model has been trained on, namely Dates, Numbers, Ages, Units, and Medications.

## Predicted Entities



{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/CONTEXTUAL_SPELL_CHECKER/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/6.Clinical_Context_Spell_Checker.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/spellcheck_clinical_en_3.4.1_3.0_1649672133997.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use

The sample code snippet may not contain all required fields of a pipeline. In this case, you can reach out a related colab notebook containing the end-to-end pipeline and more by clicking the "Open in Colab" link above.




<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = DocumentAssembler()\
  .setInputCol("text")\
  .setOutputCol("document")

tokenizer = Tokenizer()\
      .setInputCols(["document"])\
      .setOutputCol("token")\
      .setContextChars(["*", "-", "“", "(", "[", "\n", ".","\"", "”", ",", "?", ")", "]", "!", ";", ":", "'s", "’s"])

spellModel = ContextSpellCheckerModel\
    .pretrained('spellcheck_clinical', 'en', 'clinical/models')\
    .setInputCols("token")\
    .setOutputCol("checked")

pipeline = Pipeline(stages = [documentAssembler, tokenizer, spellModel])

empty = spark.createDataFrame([[""]]).toDF("text")
lp = LightPipeline(pipeline.fit(empty))

example = ["Witth the hell of phisical terapy the patient was imbulated and on postoperative, the impatient tolerating a post curgical soft diet.",
           "With paint wel controlled on orall pain medications, she was discharged too reihabilitation facilitay.",
           "Abdomen is sort, nontender, and nonintended.",
           "Patient not showing pain or any wealth problems.",
           "No cute distress"]

lp.annotate(example)
```
```scala
val assembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

val tokenizer = new Tokenizer()
      .setInputCols(Array("document"))
      .setOutputCol("token")
      .setContextChars(Array("*", "-", "“", "(", "[", "\n", ".","\"", "”", ",", "?", ")", "]", "!", ";", ":", "'s", "’s"))

 val spellChecker = ContextSpellCheckerModel.
      pretrained("spellcheck_clinical", "en", "clinical/models").
      setInputCols("token").
      setOutputCol("checked")

 val pipeline =  new Pipeline().setStages(Array(assembler, tokenizer, spellChecker))
 val empty_df = Seq("").toDF("text")
 val lp = new LightPipeline(pipeline.fit(empty_df))
 val text = Array("Witth the hell of phisical terapy the patient was imbulated and on postoperative, the impatient tolerating a post curgical soft diet.",
           "With paint wel controlled on orall pain medications, she was discharged too reihabilitation facilitay.",
           "Abdomen is sort, nontender, and nonintended.",
           "Patient not showing pain or any wealth problems.",
           "No cute distress")
 lp.annotate(text)
```
</div>

## Results

```bash
[{'checked': ['With','the','cell','of','physical','therapy','the','patient','was','ambulated','and','on','postoperative',',','the','patient','tolerating','a','post','surgical','soft','diet','.'],
  'document': ['Witth the hell of phisical terapy the patient was imbulated and on postoperative, the impatient tolerating a post curgical soft diet.'],
  'token': ['Witth','the','hell','of','phisical','terapy','the','patient','was','imbulated','and','on','postoperative',',','the','impatient','tolerating','a','post','curgical','soft','diet','.']},
 
 {'checked': ['With','pain','well','controlled','on','oral','pain','medications',',','she','was','discharged','to','rehabilitation','facility','.'],
  'document': ['With paint wel controlled on orall pain medications, she was discharged too reihabilitation facilitay.'],
  'token': ['With','paint','wel','controlled','on','orall','pain','medications',',','she','was','discharged','too','reihabilitation','facilitay','.']},
 
 {'checked': ['Abdomen','is','soft',',','nontender',',','and','nondistended','.'],
  'document': ['Abdomen is sort, nontender, and nonintended.'],
  'token': ['Abdomen','is','sort',',','nontender',',','and','nonintended','.']},
 
 {'checked': ['Patient','not','showing','pain','or','any','health','problems','.'],
  'document': ['Patient not showing pain or any wealth problems.'],
  'token': ['Patient','not','showing','pain','or','any','wealth','problems','.']},
 
 {'checked': ['No', 'acute', 'distress'],
  'document': ['No cute distress'],
  'token': ['No', 'cute', 'distress']}]
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|spellcheck_clinical|
|Compatibility:|Spark NLP for Healthcare 3.4.1+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[token]|
|Output Labels:|[corrected]|
|Language:|en|
|Size:|141.2 MB|

## References

MTSamples, i2b2 clinical notes, and several specific medical corpora.
