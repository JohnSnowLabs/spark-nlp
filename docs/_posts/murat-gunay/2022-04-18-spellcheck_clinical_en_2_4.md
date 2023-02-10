---
layout: model
title: Medical Spell Checker
author: John Snow Labs
name: spellcheck_clinical
date: 2022-04-18
tags: [spellcheck, medical, medical_spellchecker, spell_checker, spelling_corrector, en, licensed, clinical]
task: Spell Check
language: en
edition: Healthcare NLP 3.4.2
spark_version: 2.4
supported: true
annotator: SpellCheckModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---


## Description


Contextual Spell Checker is a sequence-to-sequence model that detects and corrects spelling errors in your medical input text. It’s based on Levenshtein Automation for generating candidate corrections and a Neural Language Model for ranking corrections. This model has been trained in a dataset containing data from different sources; MTSamples, i2b2 clinical notes, and several specific medical corpora. You can download the model that comes fully pretrained and ready to use. However, you can still customize it further without the need for re-training a new model from scratch. This can be accomplished by providing custom definitions for the word classes the model has been trained on, namely Dates, Numbers, Ages, Units, and Medications. This model is trained for PySpark 2.4.x users with SparkNLP 3.4.2 and above.


## Predicted Entities


{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/CONTEXTUAL_SPELL_CHECKER/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/CONTEXTUAL_SPELL_CHECKER.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/spellcheck_clinical_en_3.4.2_2.4_1650288379214.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/spellcheck_clinical_en_3.4.2_2.4_1650288379214.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}


## How to use


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

pipeline = Pipeline(stages = [		
			documentAssembler, 
			tokenizer, 
			spellModel])

light_pipeline = LightPipeline(pipeline.fit(spark.createDataFrame([[""]]).toDF("text")))

example = ["Witth the hell of phisical terapy the patient was imbulated and on postoperative, the impatient tolerating a post curgical soft diet.",
"With paint wel controlled on orall pain medications, she was discharged too reihabilitation facilitay.",
"Abdomen is sort, nontender, and nonintended.",
"Patient not showing pain or any wealth problems.",
"No cute distress"]

result = light_pipeline.annotate(example)
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

val pipeline =  new Pipeline().setStages(Array(
					assembler, 
					tokenizer, 
					spellChecker))

val light_pipeline = new LightPipeline(pipeline.fit(Seq("").toDS.toDF("text")))

val text = Array("Witth the hell of phisical terapy the patient was imbulated and on postoperative, the impatient tolerating a post curgical soft diet.",
"With paint wel controlled on orall pain medications, she was discharged too reihabilitation facilitay.",
"Abdomen is sort, nontender, and nonintended.",
"Patient not showing pain or any wealth problems.",
"No cute distress")
val result = light_pipeline.annotate(text)
```


{:.nlu-block}
```python
import nlu
nlu.load("en.spell.clinical").predict(""")

pipeline = Pipeline(stages = [		
			documentAssembler, 
			tokenizer, 
			spellModel])

light_pipeline = LightPipeline(pipeline.fit(spark.createDataFrame([[""")
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
|Compatibility:|Healthcare NLP 3.4.2+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[token]|
|Output Labels:|[corrected]|
|Language:|en|
|Size:|141.2 MB|


## References


MTSamples, i2b2 clinical notes, and several specific medical corpora.
<!--stackedit_data:
eyJoaXN0b3J5IjpbMTE5NDY1NDMsLTUwNDA1MTAzOSwtMTExNz
U5NjczMSwtNzEwNDA5MTg1XX0=
-->
