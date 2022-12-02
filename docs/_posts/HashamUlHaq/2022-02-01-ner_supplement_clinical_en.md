---
layout: model
title: Extract conditions and benefits from drug reviews
author: John Snow Labs
name: ner_supplement_clinical
date: 2022-02-01
tags: [licensed, ner, en, clinical]
task: Named Entity Recognition
language: en
edition: Healthcare NLP 3.3.4
spark_version: 3.0
supported: true
annotator: MedicalNerModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---


## Description


This model is trained to extract benefits of using drugs for certain conditions.


## Predicted Entities


`CONDITION`, `BENEFIT`


{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/1.Clinical_Named_Entity_Recognition_Model.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_supplement_clinical_en_3.3.4_3.0_1643674915917.zip){:.button.button-orange.button-orange-trans.arr.button-icon}


## How to use






<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
documentAssembler = DocumentAssembler()\
.setInputCol("text")\
.setOutputCol("document")

sentenceDetector = SentenceDetectorDLModel.pretrained("sentence_detector_dl_healthcare","en","clinical/models")\
.setInputCols(["document"])\
.setOutputCol("sentence")

tokenizer = Tokenizer()\
.setInputCols(["sentence"])\
.setOutputCol("token")\

embeddings = WordEmbeddingsModel.pretrained('embeddings_clinical', 'en', 'clinical/models') \
.setInputCols(['sentence', 'token']) \
.setOutputCol('embeddings')

ner = MedicalNerModel.pretrained('ner_supplement_clinical', 'en', 'clinical/models') \
.setInputCols(["sentence", "token", "embeddings"]) \
.setOutputCol("ner_tags")

ner_converter = NerConverter() \
.setInputCols(["sentence", "token", "ner_tags"]) \
.setOutputCol("ner_chunk")\

ner_pipeline = Pipeline(
stages = [
documentAssembler,
sentenceDetector,
tokenizer,
embeddings,
ner,
ner_converter
])

sample_df = spark.createDataFrame([["Excellent!. The state of health improves, nervousness disappears, and night sleep improves. It also promotes hair and nail growth. I recommend :)"]]).toDF("text")

result = ner_pipeline.fit(sample_df).transform(sample_df)
```
```scala
val documentAssembler = new DocumentAssembler()
.setInputCol("text")
.setOutputCol("document")

val sentenceDetector = SentenceDetectorDLModel.pretrained("sentence_detector_dl_healthcare","en","clinical/models")
.setInputCols(Array("document"))
.setOutputCol("sentence")

val tokenizer = new Tokenizer()
.setInputCols(Array("sentence"))
.setOutputCol("token")

val embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models") 
.setInputCols(Array("sentence", "token")) 
.setOutputCol("embeddings")

val ner = MedicalNerModel.pretrained("ner_supplement_clinical", "en", "clinical/models") 
.setInputCols(Array("sentence", "token", "embeddings")) 
.setOutputCol("ner_tags")

val ner_converter = new NerConverter() 
.setInputCols(Array("sentence", "token", "ner_tags")) 
.setOutputCol("ner_chunk")

val ner_pipeline = new Pipeline().setStages(Array(documentAssembler, sentenceDetector, tokenizer, embeddings, ner, ner_converter))

val sample_df = Seq("""Excellent!. The state of health improves, nervousness disappears, and night sleep improves. It also promotes hair and nail growth. I recommend :)""").toDS.toDF("text")

val result = ner_pipeline.fit(sample_df).transform(sample_df)
```


{:.nlu-block}
```python
import nlu
nlu.load("en.med_ner.supplement_clinical").predict("""Excellent!. The state of health improves, nervousness disappears, and night sleep improves. It also promotes hair and nail growth. I recommend :)""")
```

</div>


## Results


```bash
+------------------------+---------------+
| chunk                  | ner_label     |
+------------------------+---------------+
| nervousness            | CONDITION     |
| night sleep improves   | BENEFIT       |
| hair                   | BENEFIT       |
| nail                   | BENEFIT       |
+------------------------+---------------+


```


{:.model-param}
## Model Information


{:.table-model}
|---|---|
|Model Name:|ner_supplement_clinical|
|Compatibility:|Healthcare NLP 3.3.4+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|en|
|Size:|14.5 MB|


## References


Trained on healthsea dataset: https://github.com/explosion/healthsea/tree/main/project/assets/ner


## Benchmarking


```bash
label	       tp	 fp	 fn	 prec	 		rec	 		f1
B-BENEFIT	   268	 39	 42	 0.87296414	 0.86451614	 0.86871964
I-CONDITION	   178	 29	 72	 0.8599034	 0.712	 	 0.7789934
I-BENEFIT	   52	 14	 32	 0.7878788	 0.61904764	 0.6933334
B-CONDITION	   365	 78	 61	 0.82392776	 0.85680753	 0.840046
Macro-average  863  160  207 0.8361685   0.7630928   0.7979612
Micro-average  863  160  207 0.84359723  0.80654204  0.8246535
```
<!--stackedit_data:
eyJoaXN0b3J5IjpbMjEwOTU1MjcwNl19
-->