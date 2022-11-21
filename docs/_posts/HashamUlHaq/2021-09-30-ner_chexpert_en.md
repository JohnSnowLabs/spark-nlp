---
layout: model
title: Detect Anatomical and Observation Entities in Chest Radiology Reports (CheXpert)
author: John Snow Labs
name: ner_chexpert
date: 2021-09-30
tags: [licensed, ner, clinical, en]
task: Named Entity Recognition
language: en
edition: Healthcare NLP 3.3.0
spark_version: 3.0
supported: true
annotator: MedicalNerModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---


## Description


This model extracts `Anatomical` and `Observation` entities from Chest Radiology Reports.


## Predicted Entities


`ANAT - Anatomy`, `OBS - Observation`


{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/NER_RADIOLOGY/){:.button.button-orange}{:target="_blank"}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/1.Clinical_Named_Entity_Recognition_Model.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_chexpert_en_3.3.0_3.0_1633010671460.zip){:.button.button-orange.button-orange-trans.arr.button-icon}


## How to use






<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
document_assembler = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")
         
sentence_detector = SentenceDetector()\
    .setInputCols(["document"])\
    .setOutputCol("sentence")

tokenizer = Tokenizer()\
    .setInputCols(["sentence"])\
    .setOutputCol("token")

embeddings_clinical = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")\
    .setInputCols(["sentence", "token"])\
    .setOutputCol("embeddings")

clinical_ner = MedicalNerModel.pretrained("ner_chexpert", "en", "clinical/models")\
    .setInputCols(["sentence", "token", "embeddings"])\
    .setOutputCol("ner")

ner_converter = NerConverter()\
 	.setInputCols(["sentence", "token", "ner"])\
 	.setOutputCol("ner_chunk")

nlpPipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, embeddings_clinical, clinical_ner, ner_converter])

model = nlpPipeline.fit(spark.createDataFrame([[""]]).toDF("text"))

results = model.transform(spark.createDataFrame([["FINAL REPORT HISTORY : Chest tube leak , to assess for pneumothorax. FINDINGS : In comparison with study of ___ , the endotracheal tube and Swan - Ganz catheter have been removed . The left chest tube remains in place and there is no evidence of pneumothorax. Mild atelectatic changes are seen at the left base."]], ["text"]))

```
```scala
val document_assembler = new DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document")
         
val sentence_detector = new SentenceDetector()
    .setInputCols("document")
    .setOutputCol("sentence")

val tokenizer = new Tokenizer()
    .setInputCols("sentence")
    .setOutputCol("token")

val embeddings_clinical = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")
    .setInputCols(Array("sentence", "token"))
    .setOutputCol("embeddings")

val ner = MedicalNerModel.pretrained("ner_chexpert", "en", "clinical/models")
    .setInputCols(Array("sentence", "token", "embeddings"))
    .setOutputCol("ner")

val ner_converter = new NerConverter()
 	.setInputCols(Array("sentence", "token", "ner"))
 	.setOutputCol("ner_chunk")

val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, tokenizer, embeddings_clinical, ner, ner_converter))

val data = Seq("""FINAL REPORT HISTORY : Chest tube leak , to assess for pneumothorax. FINDINGS : In comparison with study of ___ , the endotracheal tube and Swan - Ganz catheter have been removed . The left chest tube remains in place and there is no evidence of pneumothorax. Mild atelectatic changes are seen at the left base.""").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("en.med_ner.chexpert").predict("""FINAL REPORT HISTORY : Chest tube leak , to assess for pneumothorax. FINDINGS : In comparison with study of ___ , the endotracheal tube and Swan - Ganz catheter have been removed . The left chest tube remains in place and there is no evidence of pneumothorax. Mild atelectatic changes are seen at the left base.""")
```

</div>


## Results


```bash
|    | chunk                    | label   |
|---:|:-------------------------|:--------|
|  0 | endotracheal tube        | OBS     |
|  1 | Swan - Ganz catheter     | OBS     |
|  2 | left chest               | ANAT    |
|  3 | tube                     | OBS     |
|  4 | in place                 | OBS     |
|  5 | pneumothorax             | OBS     |
|  6 | Mild atelectatic changes | OBS     |
|  7 | left base                | ANAT    |
```


{:.model-param}
## Model Information


{:.table-model}
|---|---|
|Model Name:|ner_chexpert|
|Compatibility:|Healthcare NLP 3.3.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|en|


## Data Source


Trained on CheXpert dataset explain in https://arxiv.org/pdf/2106.14463.pdf.


## Benchmarking


```bash
label	        tp	   fp    fn	   prec        rec	       f1
I-ANAT_DP	    26	   11    11	   0.7027027   0.7027027   0.7027027
B-OBS_DP	    1489   141	 104   0.9134969   0.9347144   0.9239839
I-OBS_DP	    16	   3     54	   0.84210527  0.22857143  0.35955057
B-ANAT_DP	    1125   39    45	   0.96649486  0.96153843  0.96401024
Macro-average	2656   194   214   0.8561999   0.70688176  0.7744088
Micro-average	2656   194   214   0.9319298   0.92543554  0.9286713
```
<!--stackedit_data:
eyJoaXN0b3J5IjpbNTUxMDE4NTVdfQ==
-->