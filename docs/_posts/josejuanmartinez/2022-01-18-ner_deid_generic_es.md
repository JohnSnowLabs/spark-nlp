---
layout: model
title: Detect PHI for Deidentification purposes (Spanish, reduced entities)
author: John Snow Labs
name: ner_deid_generic
date: 2022-01-18
tags: [deid, es, licensed]
task: De-identification
language: es
edition: Healthcare NLP 3.3.4
spark_version: 3.0
supported: true
annotator: MedicalNerModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---


## Description


Named Entity Recognition annotators allow for a generic model to be trained by using a Deep Learning architecture (Char CNNs - BiLSTM - CRF - word embeddings) inspired on a former state of the art model for NER: Chiu & Nicols, Named Entity Recognition with Bidirectional LSTM,CNN. 


Deidentification NER (Spanish) is a Named Entity Recognition model that annotates text to find protected health information that may need to be de-identified. It detects 7 entities. This NER model is trained with a combination of custom datasets, Spanish 2002 conLL, MeddoProf dataset and several data augmentation mechanisms and it's a reduced version of `ner_deid_subentity`.


## Predicted Entities


`CONTACT`, `NAME`, `DATE`, `ID`, `LOCATION`, `PROFESSION`, `AGE`


{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_deid_generic_es_3.3.4_3.0_1642528473168.zip){:.button.button-orange.button-orange-trans.arr.button-icon}


## How to use






<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
documentAssembler = nlp.DocumentAssembler()\
.setInputCol("text")\
.setOutputCol("document")

# Feel free to experiment with multilingual or Spanish nlp.SentenceDetector instead
sentenceDetector = nlp.SentenceDetectorDLModel.pretrained("sentence_detector_dl","xx")\
.setInputCols(["document"])\
.setOutputCol("sentence")

tokenizer = nlp.Tokenizer()\
.setInputCols(["sentence"])\
.setOutputCol("token")

embeddings = nlp.WordEmbeddingsModel.pretrained("embeddings_sciwiki_300d","es","clinical/models")\
	.setInputCols(["sentence","token"])\
	.setOutputCol("word_embeddings")

clinical_ner = medical.NerModel.pretrained("ner_deid_generic", "es", "clinical/models")\
.setInputCols(["sentence","token","word_embeddings"])\
.setOutputCol("ner")

nlpPipeline = Pipeline(stages=[
documentAssembler,
sentenceDetector,
tokenizer,
embeddings,
clinical_ner])

text = ['''
Antonio Pérez Juan, nacido en Cadiz, España. Aún no estaba vacunado, se infectó con Covid-19 el dia 14/03/2020 y tuvo que ir al Hospital. Fue tratado con anticuerpos monoclonales en la Clinica San Carlos.
''']

df = spark.createDataFrame([text]).toDF("text")

results = nlpPipeline.fit(df).transform(df)
```
```scala
val documentAssembler = new DocumentAssembler()
.setInputCol("text")
.setOutputCol("document")

val sentenceDetector = SentenceDetectorDLModel.pretrained("sentence_detector_dl","xx")
.setInputCols(Array("document"))
.setOutputCol("sentence")

val tokenizer = new Tokenizer()
.setInputCols(Array("sentence"))
.setOutputCol("token")

embeddings = WordEmbeddingsModel.pretrained("embeddings_sciwiki_300d","es","clinical/models")
	.setInputCols(Array("sentence","token"))
	.setOutputCol("word_embeddings")

clinical_ner = MedicalNerModel.pretrained("ner_deid_generic", "es", "clinical/models")
.setInputCols(Array("sentence","token","word_embeddings"))
.setOutputCol("ner")

val pipeline = new Pipeline().setStages(Array(documentAssembler, sentenceDetector, tokenizer, embeddings, clinical_ner))

val text = """Antonio Pérez Juan, nacido en Cadiz, España. Aún no estaba vacunado, se infectó con Covid-19 el dia 14/03/2020 y tuvo que ir al Hospital. Fue tratado con anticuerpos monoclonales en la Clinica San Carlos."""

val df = Seq(text).toDS.toDF("text")

val results = pipeline.fit(df).transform(df)
```


{:.nlu-block}
```python
import nlu
nlu.load("es.med_ner.deid.generic").predict("""
Antonio Pérez Juan, nacido en Cadiz, España. Aún no estaba vacunado, se infectó con Covid-19 el dia 14/03/2020 y tuvo que ir al Hospital. Fue tratado con anticuerpos monoclonales en la Clinica San Carlos.
""")
```

</div>


## Results


```bash
+------------+----------+
|       token| ner_label|
+------------+----------+
|     Antonio|    B-NAME|
|       Pérez|    I-NAME|
|        Juan|    I-NAME|
|           ,|         O|
|      nacido|         O|
|          en|         O|
|       Cadiz|B-LOCATION|
|           ,|         O|
|      España|B-LOCATION|
|           .|         O|
|         Aún|         O|
|          no|         O|
|      estaba|         O|
|    vacunado|         O|
|           ,|         O|
|          se|         O|
|     infectó|         O|
|         con|         O|
|    Covid-19|         O|
|          el|         O|
|         dia|         O|
|  14/03/2020|    B-DATE|
|           y|         O|
|        tuvo|         O|
|         que|         O|
|          ir|         O|
|          al|         O|
|    Hospital|         O|
|         Fue|         O|
|     tratado|         O|
|         con|         O|
| anticuerpos|         O|
|monoclonales|         O|
|          en|         O|
|          la|         O|
|     Clinica|B-LOCATION|
|         San|I-LOCATION|
|      Carlos|I-LOCATION|
|           .|         O|
+------------+----------+
```


{:.model-param}
## Model Information


{:.table-model}
|---|---|
|Model Name:|ner_deid_generic|
|Compatibility:|Healthcare NLP 3.3.4+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, word_embeddings]|
|Output Labels:|[ner]|
|Language:|es|
|Size:|15.0 MB|
|Dependencies:|embeddings_sciwiki_300d|


## Data Source


- Internal JSL annotated corpus
- [Spanish conLL](https://www.clips.uantwerpen.be/conll2002/ner/data/)
- [MeddoProf](https://temu.bsc.es/meddoprof/data/)


## Benchmarking


```bash
label      tp     fp     fn   total  precision  recall      f1
CONTACT   166.0    9.0    8.0   174.0     0.9486   0.954  0.9513
NAME  2879.0  195.0  191.0  3070.0     0.9366  0.9378  0.9372
DATE  1839.0   29.0   18.0  1857.0     0.9845  0.9903  0.9874
ID   119.0   11.0   12.0   131.0     0.9154  0.9084  0.9119
LOCATION  5149.0  711.0  607.0  5756.0     0.8787  0.8945  0.8865
PROFESSION   236.0   49.0  168.0   404.0     0.8281  0.5842  0.6851
AGE   313.0   33.0   50.0   363.0     0.9046  0.8623  0.8829
macro     -      -      -       -         -       -     0.891749
micro     -      -      -       -         -       -     0.909897
```
