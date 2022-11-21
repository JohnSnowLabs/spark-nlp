---
layout: model
title: Medical Spell Checker Pipeline
author: John Snow Labs
name: spellcheck_clinical_pipeline
date: 2022-04-13
tags: [spellcheck, medical, medical_spell_checker, spell_corrector, spell_pipeline, en, licensed, clinical]
task: Spell Check
language: en
edition: Healthcare NLP 3.4.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---


## Description


This pretrained medical spellchecker pipeline is built on the top of [spellcheck_clinical](https://nlp.johnsnowlabs.com/2022/04/11/spellcheck_clinical_en_3_0.html) model.


{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/CONTEXTUAL_SPELL_CHECKER/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/CONTEXTUAL_SPELL_CHECKER.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/spellcheck_clinical_pipeline_en_3.4.1_3.0_1649854381549.zip){:.button.button-orange.button-orange-trans.arr.button-icon}


## How to use






<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
from sparknlp.pretrained import PretrainedPipeline

pipeline = PretrainedPipeline("spellcheck_clinical_pipeline", "en", "clinical/models")


example = ["Witth the hell of phisical terapy the patient was imbulated and on postoperative, the impatient tolerating a post curgical soft diet.",
           "With paint wel controlled on orall pain medications, she was discharged too reihabilitation facilitay.",
           "Abdomen is sort, nontender, and nonintended.",
           "Patient not showing pain or any wealth problems.",
           "No cute distress"]


pipeline.fullAnnotate(example)
```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val pipeline = new PretrainedPipeline("spellcheck_clinical_pipeline", "en", "clinical/models")


val example = Array("Witth the hell of phisical terapy the patient was imbulated and on postoperative, the impatient tolerating a post curgical soft diet.",
           "With paint wel controlled on orall pain medications, she was discharged too reihabilitation facilitay.",
           "Abdomen is sort, nontender, and nonintended.",
           "Patient not showing pain or any wealth problems.",
           "No cute distress")


pipeline.fullAnnotate(example)
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
|Model Name:|spellcheck_clinical_pipeline|
|Type:|pipeline|
|Compatibility:|Healthcare NLP 3.4.1+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|
|Size:|141.2 MB|


## Included Models


- DocumentAssembler
- TokenizerModel
- ContextSpellCheckerModel
<!--stackedit_data:
eyJoaXN0b3J5IjpbODIwMjU2NjU2LC01MzE1NDc3NjIsNzQ2OD
U3MDI0XX0=
-->