---
layout: model
title: Detect Drugs - Generalized Single Entity (ner_drugs_greedy)
author: John Snow Labs
name: ner_drugs_greedy
date: 2021-03-31
tags: [ner, clinical, licensed, en]
task: Named Entity Recognition
language: en
edition: Healthcare NLP 3.0.0
spark_version: 3.0
supported: true
annotator: MedicalNerModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This is a single entity model that generalises all posology concepts into one and finds longest available chunks of drugs. It is trained using `embeddings_clinical` so please use the same embeddings in the pipeline.

## Predicted Entities

`DRUG`.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/1.Clinical_Named_Entity_Recognition_Model.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_drugs_greedy_en_3.0.0_3.0_1617208410026.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/ner_drugs_greedy_en_3.0.0_3.0_1617208410026.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
document_assembler = DocumentAssembler()\
	.setInputCol("text")\
	.setOutputCol("document")

sentence_detector = SentenceDetectorDLModel.pretrained("sentence_detector_dl_healthcare", "en", "clinical/models") \
	.setInputCols(["document"]) \
	.setOutputCol("sentence")

tokenizer = Tokenizer()\
	.setInputCols(["sentence"])\
	.setOutputCol("token")

word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")\
    .setInputCols(["sentence", "token"])\
    .setOutputCol("embeddings")

clinical_ner = MedicalNerModel.pretrained("ner_drugs_greedy", "en", "clinical/models") \
    .setInputCols(["sentence", "token", "embeddings"]) \
    .setOutputCol("ner")

ner_converter = NerConverter() \
    .setInputCols(["sentence", "token", "ner"]) \
    .setOutputCol("entities")

nlp_pipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, word_embeddings, clinical_ner, ner_converter])

model = nlp_pipeline.fit(spark.createDataFrame([[""]]).toDF("text"))

results = model.transform(spark.createDataFrame([["DOSAGE AND ADMINISTRATION The initial dosage of hydrocortisone tablets may vary from 20 mg to 240 mg of hydrocortisone per day depending on the specific disease entity being treated."]]).toDF("text"))
```
```scala
val document_assembler = new DocumentAssembler()
	.setInputCol("text")
	.setOutputCol("document")

val sentence_detector = SentenceDetectorDLModel.pretrained("sentence_detector_dl_healthcare", "en", "clinical/models")
	.setInputCols("document")
	.setOutputCol("sentence")

val tokenizer = new Tokenizer()
	.setInputCols("sentence")
	.setOutputCol("token")

val word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")
    .setInputCols(Array("sentence", "token"))
    .setOutputCol("embeddings")

val clinical_ner = MedicalNerModel.pretrained("ner_drugs_greedy", "en", "clinical/models")
    .setInputCols(Array("sentence", "token", "embeddings"))
    .setOutputCol("ner")

val ner_converter = new NerConverter()
  .setInputCols(Array("sentence", "token", "ner"))
  .setOutputCol("entities")

val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, tokenizer, word_embeddings, clinical_ner, ner_converter))

val text = """DOSAGE AND ADMINISTRATION The initial dosage of hydrocortisone tablets may vary from 20 mg to 240 mg of hydrocortisone per day depending on the specific disease entity being treated."""

val data = Seq(text).toDS.toDF("text")

val results = pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("en.med_ner.drugsgreedy").predict("""DOSAGE AND ADMINISTRATION The initial dosage of hydrocortisone tablets may vary from 20 mg to 240 mg of hydrocortisone per day depending on the specific disease entity being treated.""")
```

</div>

## Results

```bash
+-----------------------------------+------------+
| chunk                             | ner_label  |
+-----------------------------------+------------+
| hydrocortisone tablets            | DRUG       |
| 20 mg to 240 mg of hydrocortisone | DRUG       |
+-----------------------------------+------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_drugs_greedy|
|Compatibility:|Healthcare NLP 3.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|en|

## Data Source

Trained on augmented i2b2_med7 + FDA dataset with ``embeddings_clinical``, [https://www.i2b2.org/NLP/Medication](https://www.i2b2.org/NLP/Medication).

## Benchmarking

```bash
label	 tp	 fp	 fn	 prec	 rec	 f1
I-DRUG	 37858	 4166	 3338	 0.90086615	 0.91897273	 0.9098294
B-DRUG	 29926	 2006	 1756	 0.937179	 0.9445742	 0.9408621
tp: 67784 fp: 6172 fn: 5094 labels: 2
Macro-average	 prec: 0.91902256, rec: 0.9317734, f1: 0.92535406
Micro-average	 prec: 0.916545, rec: 0.93010235, f1: 0.9232739
```