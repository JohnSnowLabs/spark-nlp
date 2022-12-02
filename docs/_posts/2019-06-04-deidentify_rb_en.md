---
layout: model
title: Deidentify RB
author: John Snow Labs
name: deidentify_rb
class: DeIdentificationModel
language: en
repository: clinical/models
date: 2019-06-04
task: De-identification
edition: Healthcare NLP 2.0.2
spark_version: 2.4
tags: [clinical,licensed,en]
supported: true
annotator: DeIdentificationModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

{:.h2_title}
## Description
Anonymization and DeIdentification model based on outputs from DeId NERs and Replacement Dictionaries.


## Predicted Entities 
Personal Information in order to deidentify.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/deidentify_rb_en_2.0.2_2.4_1559672122511.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
{:.h2_title}
## How to use 
<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
...
nlpPipeline = Pipeline(stages=[documentAssembler, sentenceDetector, tokenizer, word_embeddings, clinical_ner, ner_converter])
model = nlpPipeline.fit(spark.createDataFrame([[""]]).toDF("text"))
masker = DeIdentificationModel.pretrained("deidentify_rb","en","clinical/models")\
	.setInputCols("sentence","token","chunk")\
	.setOutputCol("deidentified")\
.setMode("mask")

text = '''A . Record date : 2093-01-13 , David Hale , M.D . , Name : Hendrickson , Ora MR . # 7194334 Date : 01/13/93 PCP : Oliveira , 25 years-old , Record date : 2079-11-09 . Cocke County Baptist Hospital . 0295 Keats Street'''
result = model.transform(spark.createDataFrame([[text]]).toDF("text"))    
deid_text = masker.transform(result)
```

```scala
...
val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, tokenizer, word_embeddings, ner, ner_converter))
val data = Seq("A . Record date : 2093-01-13 , David Hale , M.D . , Name : Hendrickson , Ora MR . # 7194334 Date : 01/13/93 PCP : Oliveira , 25 years-old , Record date : 2079-11-09 . Cocke County Baptist Hospital . 0295 Keats Street").toDF("text")
val result = pipeline.fit(data).transform(data)

val masker = DeIdentificationModel.pretrained("deidentify_rb","en","clinical/models")
.setInputCols(Array("sentence", "token", "chunk"))
.setOutputCol("deidentified")
.setMode("mask")
val deid_text = new masker.transform(result)

```


{:.nlu-block}
```python
import nlu
nlu.load("en.de_identify").predict("""A . Record date : 2093-01-13 , David Hale , M.D . , Name : Hendrickson , Ora MR . # 7194334 Date : 01/13/93 PCP : Oliveira , 25 years-old , Record date : 2079-11-09 . Cocke County Baptist Hospital . 0295 Keats Street""")
```

</div>

{:.h2_title}
## Results
```bash
|   | sentence                                                                              | deidentified                                                                |
|---|---------------------------------------------------------------------------------------|-----------------------------------------------------------------------------|
| 0 | A .                                                                                   | A .                                                                         |
| 1 | Record date : 2093-01-13 , David Hale , M.D .                                         | Record date : <DATE> , David Hale , M.D .                                   |
| 2 | , Name : Hendrickson , Ora MR .                                                       | , Name : Hendrickson , Ora MR .                                             |
| 3 | # 7194334 Date : 01/13/93 PCP : Oliveira , 25 years-old , Record date : 2079-11-09 .  | # <ID> Date : <DATE> PCP : Oliveira , 25 years-old , Record date : <DATE> . |
| 4 | Cocke County Baptist Hospital .                                                       | Cocke County Baptist Hospital .                                             |
| 5 | 0295 Keats Street                                                                     | <ID> Keats Street                                                           |
```

{:.model-param}
## Model Information

{:.table-model}
|---------------|------------------------|
| Name:          | deidentify_rb          |
| Type:   | DeIdentificationModel  |
| Compatibility: | Spark NLP 2.0.2+                  |
| License:       | Licensed               |
| Edition:       | Official             |
|Input labels:        | [document, token, chunk] |
|Output labels:       | [document]               |
| Language:      | en                     |
| Dependencies: | ner_deid               |

{:.h2_title}
## Data Source
Rule based DeIdentifier based on `ner_deid`.
