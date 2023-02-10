---
layout: model
title: Detect PHI for Deidentification purposes (Portuguese)
author: John Snow Labs
name: ner_deid_subentity
date: 2022-04-13
tags: [deid, deidentification, pt, licensed]
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


Deidentification NER (Portuguese) is a Named Entity Recognition model that annotates text to find protected health information that may need to be de-identified. It detects 15 entities.


This NER model is trained with a combination of custom datasets with several data augmentation mechanisms.


## Predicted Entities


`PATIENT`, `HOSPITAL`, `DATE`, `ORGANIZATION`, `CITY`, `ID`, `STREET`, `SEX`, `EMAIL`, `ZIP`, `PROFESSION`, `PHONE`, `COUNTRY`, `DOCTOR`, `AGE`


{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/DEID_PHI_TEXT_MULTI/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/DEID_PHI_TEXT_MULTI.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_deid_subentity_pt_3.4.2_3.0_1649840643338.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/ner_deid_subentity_pt_3.4.2_3.0_1649840643338.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}


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


clinical_ner = medical.NerModel.pretrained("ner_deid_subentity", "pt", "clinical/models")\
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

clinical_ner = MedicalNerModel.pretrained("ner_deid_subentity", "pt", "clinical/models")
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

val data = Seq(text).toDF("text")

val results = pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("pt.med_ner.deid.subentity").predict("""
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
+-----------------+------------+
|chunk            |ner_label   |
+-----------------+------------+
|Pedro Gonçalves  |PATIENT     |
|2569870          |ID          |
|Rua Das Flores 23|STREET      |
|Porto            |ORGANIZATION|
|21754-987        |ID          |
|10/10/1963       |DATE        |
|53               |AGE         |
|17/06/2016       |DATE        |
|Maria Santos     |DOCTOR      |
+-----------------+------------+
```


{:.model-param}
## Model Information


{:.table-model}
|---|---|
|Model Name:|ner_deid_subentity|
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
label      		tp     	fp     	fn   	total  	precision  	recall      f1
PATIENT  		2142.0  186.0   59.0  	2201.0 	0.9201  	0.9732  	0.9459
HOSPITAL   		248.0   10.0   	46.0   	294.0   0.9612  	0.8435  	0.8986
DATE  			1306.0	26.0   	15.0  	1321.0  0.9805  	0.9886  	0.9845
ORGANIZATION	3038.0	31.0  	156.0  	3194.0  0.9899  	0.9512  	0.9701
CITY  			1836.0	58.0   	15.0  	1851.0  0.9694  	0.9919  	0.9805
ID    			56.0    8.0    	7.0    	63.0	0.875  		0.8889  	0.8819
STREET   		155.0	0.0    	0.0   	155.0	1.0     	1.0     	1.0
SEX   			658.0   20.0   	19.0   	677.0	0.9705  	0.9719  	0.9712
EMAIL   		131.0	0.0    	0.0   	131.0	1.0     	1.0     	1.0
ZIP   			125.0	2.0    	0.0   	125.0	0.9843     	1.0  		0.9921
PROFESSION   	237.0   15.0   	39.0   	276.0	0.9405  	0.8587  	0.8977
PHONE    		64.0    2.0    	0.0    	64.0	0.9697     	1.0  		0.9846
COUNTRY   		502.0   27.0   	74.0   	576.0	0.949  		0.8715  	0.9086
DOCTOR   		461.0   35.0   	31.0   	492.0	0.9294   	0.937  		0.9332
AGE   			538.0   17.0    8.0   	546.0	0.9694  	0.9853  	0.9773
macro      		-       -      	-       -      	-       	-  			0.9551
micro      		-       -      	-       -      	-       	-  			0.9619
```
