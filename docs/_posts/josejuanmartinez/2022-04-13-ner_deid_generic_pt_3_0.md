---
layout: model
title: Detect PHI for Deidentification purposes (Portuguese, reduced entities)
author: John Snow Labs
name: ner_deid_generic
date: 2022-04-13
tags: [deid, deidentification, pt, licensed, clinical]
task: De-identification
language: pt
edition: Healthcare NLP 3.4.2
spark_version: 3.0
supported: true
annotator: MedicalNerModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Named Entity Recognition annotators allow for a generic model to be trained by using a Deep Learning architecture (Char CNNs - BiLSTM - CRF - word embeddings) inspired on a former state of the art model for NER: Chiu & Nicols, Named Entity Recognition with Bidirectional LSTM,CNN. 

Deidentification NER (Portuguese) is a Named Entity Recognition model that annotates text to find protected health information that may need to be de-identified. It detects 7 entities. This NER model is trained with a combination of custom datasets and data augmentation techniques.

## Predicted Entities

`CONTACT`, `NAME`, `DATE`, `ID`, `SEX`, `LOCATION`, `PROFESSION`, `AGE`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/DEID_PHI_TEXT_MULTI/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/DEID_PHI_TEXT_MULTI.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_deid_generic_pt_3.4.2_3.0_1649846957944.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/ner_deid_generic_pt_3.4.2_3.0_1649846957944.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
documentAssembler = nlp.DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

sentenceDetector = nlp.SentenceDetectorDLModel.pretrained("sentence_detector_dl","xx")\
    .setInputCols(["document"])\
    .setOutputCol("sentence")


tokenizer = nlp.Tokenizer()\
    .setInputCols(["sentence"])\
    .setOutputCol("token")


embeddings = nlp.WordEmbeddingsModel.pretrained("w2v_cc_300d", "pt")\
    .setInputCols(["sentence","token"])\
    .setOutputCol("word_embeddings")


clinical_ner = medical.NerModel.pretrained("ner_deid_generic", "pt", "clinical/models")\
    .setInputCols(["sentence","token","word_embeddings"])\
    .setOutputCol("ner")

ner_converter = nlp.NerConverter()\
    .setInputCols(["sentence","token","ner"])\
    .setOutputCol("ner_chunk")

nlpPipeline = Pipeline(stages=[
    documentAssembler,
    sentenceDetector,
    tokenizer,
    embeddings,
    clinical_ner,
    ner_converter])

text = ['''
Detalhes do paciente.
Nome do paciente:  Pedro Gonçalves
NHC: 2569870.
Endereço: Rua Das Flores 23.
Cidade/ Província: Porto.
Código Postal: 21754-987.
Dados de cuidados.
Data de nascimento: 10/10/1963.
Idade: 53 anos Sexo: Homen
Data de admissão: 17/06/2016.
Doutora: Maria Santos
''']

data = spark.createDataFrame([text]).toDF("text")

result = nlpPipeline.fit(data).transform(data)
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

embeddings = WordEmbeddingsModel.pretrained("w2v_cc_300d", "pt")
	.setInputCols(Array("sentence","token"))
	.setOutputCol("word_embeddings")

clinical_ner = MedicalNerModel.pretrained("ner_deid_generic", "pt", "clinical/models")
	.setInputCols(Array("sentence","token","word_embeddings"))
	.setOutputCol("ner")

val ner_converter = new NerConverter()
	.setInputCols(Array("sentence", "token", "ner"))
	.setOutputCol("ner_chunk")

val pipeline = new Pipeline().setStages(Array(documentAssembler, sentenceDetector, tokenizer, embeddings, clinical_ner, ner_converter))

val text = """Detalhes do paciente.
Nome do paciente:  Pedro Gonçalves
NHC: 2569870.
Endereço: Rua Das Flores 23.
Cidade/ Província: Porto.
Código Postal: 21754-987.
Dados de cuidados.
Data de nascimento: 10/10/1963.
Idade: 53 anos Sexo: Homen
Data de admissão: 17/06/2016.
Doutora: Maria Santos"""

val df = Seq(text).toDF("text")

val results = pipeline.fit(df).transform(df)
```


{:.nlu-block}
```python
import nlu
nlu.load("pt.med_ner.deid.generic").predict("""
Detalhes do paciente.
Nome do paciente:  Pedro Gonçalves
NHC: 2569870.
Endereço: Rua Das Flores 23.
Cidade/ Província: Porto.
Código Postal: 21754-987.
Dados de cuidados.
Data de nascimento: 10/10/1963.
Idade: 53 anos Sexo: Homen
Data de admissão: 17/06/2016.
Doutora: Maria Santos
""")
```

</div>


## Results


```bash
+-----------------+---------+
|chunk            |ner_label|
+-----------------+---------+
|Pedro Gonçalves  |NAME     |
|2569870          |ID       |
|Rua Das Flores 23|LOCATION |
|Porto            |LOCATION |
|21754-987        |LOCATION |
|10/10/1963       |DATE     |
|53               |AGE      |
|17/06/2016       |DATE     |
|Maria Santos     |NAME     |
+-----------------+---------+
```


{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_deid_generic|
|Compatibility:|Healthcare NLP 3.4.2+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|pt|
|Size:|15.0 MB|

## References

- Custom John Snow Labs datasets
- Data augmentation techniques

## Benchmarking


```bash
label      	tp     	fp     fn     total    precision  recall   f1 
CONTACT   	191.0   2.0    2.0    193.0    0.9896     0.9896   0.9896 
NAME  		2640.0  82.0   52.0   2692.0   0.9699     0.9807   0.9752 
DATE  		1316.0  24.0   5.0	  1321.0   0.9821     0.9962   0.9891 
ID    		54.0   	3.0    9.0    63.0     0.9474     0.8571      0.9 
SEX   		669.0   9.0    8.0    677.0    0.9867     0.9882   0.9875 
LOCATION  	5784.0	149.0  206.0  5990.0   0.9749     0.9656   0.9702 
PROFESSION	249.0   17.0   27.0   276.0    0.9361     0.9022   0.9188 
AGE   		536.0   14.0   10.0   546.0    0.9745     0.9817   0.9781 
macro       -      	-      -       -         -        -        0.9636 
macro       -      	-      -       -         -        -        0.9736 
```