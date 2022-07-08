---
layout: model
title: Detect PHI for Deidentification purposes (Portuguese, reduced entities)
author: John Snow Labs
name: ner_deid_generic
date: 2022-04-13
tags: [deid, deidentification, pt, licensed, clinical]
task: De-identification
language: pt
edition: Spark NLP for Healthcare 3.4.2
spark_version: 3.0
supported: true
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
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_deid_generic_pt_3.4.2_3.0_1649846957944.zip){:.button.button-orange.button-orange-trans.arr.button-icon}


## How to use






<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
        .setInputCol("text")\
        .setOutputCol("document")
        
sentenceDetector = SentenceDetectorDLModel.pretrained("sentence_detector_dl","xx")\
        .setInputCols(["document"])\
        .setOutputCol("sentence")


tokenizer = Tokenizer()\
        .setInputCols(["sentence"])\
        .setOutputCol("token")


embeddings = WordEmbeddingsModel.pretrained("w2v_cc_300d", "pt")\
	.setInputCols(["sentence","token"])\
	.setOutputCol("word_embeddings")


clinical_ner = MedicalNerModel.pretrained("ner_deid_generic", "pt", "clinical/models")\
        .setInputCols(["sentence","token","word_embeddings"])\
        .setOutputCol("ner")


nlpPipeline = Pipeline(stages=[
        documentAssembler,
        sentenceDetector,
        tokenizer,
        embeddings,
        clinical_ner])


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


df = spark.createDataFrame([text]).toDF("text")


results = nlpPipeline.fit(data).transform(data)
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


val pipeline = new Pipeline().setStages(Array(documentAssembler, sentenceDetector, tokenizer, embeddings, clinical_ner))


val text = "Detalhes do paciente.
Nome do paciente:  Pedro Gonçalves
NHC: 2569870.
Endereço: Rua Das Flores 23.
Cidade/ Província: Porto.
Código Postal: 21754-987.
Dados de cuidados.
Data de nascimento: 10/10/1963.
Idade: 53 anos Sexo: Homen
Data de admissão: 17/06/2016.
Doutora: Maria Santos"


val df = Seq(text).toDF("text")


val results = pipeline.fit(data).transform(data)
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
----------+----------+
|     token| ner_label|
+----------+----------+
|  Detalhes|         O|
|        do|         O|
|  paciente|         O|
|         .|         O|
|      Nome|         O|
|        do|         O|
|  paciente|         O|
|         :|         O|
|     Pedro|    B-NAME|
| Gonçalves|    I-NAME|
|       NHC|         O|
|         :|         O|
|   2569870|      B-ID|
|         .|         O|
|  Endereço|         O|
|         :|         O|
|       Rua|B-LOCATION|
|       Das|I-LOCATION|
|    Flores|I-LOCATION|
|        23|I-LOCATION|
|         .|         O|
|   Cidade/|         O|
| Província|         O|
|         :|         O|
|     Porto|B-LOCATION|
|         .|         O|
|    Código|         O|
|    Postal|         O|
|         :|         O|
| 21754-987|B-LOCATION|
|         .|         O|
|     Dados|         O|
|        de|         O|
|  cuidados|         O|
|         .|         O|
|      Data|         O|
|        de|         O|
|nascimento|         O|
|         :|         O|
|10/10/1963|    B-DATE|
|         .|         O|
|     Idade|         O|
|         :|         O|
|        53|     B-AGE|
|      anos|         O|
|      Sexo|         O|
|         :|         O|
|     Homen|         O|
|      Data|         O|
|        de|         O|
|  admissão|         O|
|         :|         O|
|17/06/2016|    B-DATE|
|         .|         O|
|   Doutora|         O|
|         :|         O|
|     Maria|    B-NAME|
|    Santos|    I-NAME|
+----------+----------+
```


{:.model-param}
## Model Information


{:.table-model}
|---|---|
|Model Name:|ner_deid_generic|
|Compatibility:|Spark NLP for Healthcare 3.4.2+|
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
     label      tp     fp     fn   total  precision  recall      f1 
   CONTACT   191.0    2.0    2.0   193.0     0.9896  0.9896  0.9896 
      NAME  2640.0   82.0   52.0  2692.0     0.9699  0.9807  0.9752 
      DATE  1316.0   24.0    5.0  1321.0     0.9821  0.9962  0.9891 
        ID    54.0    3.0    9.0    63.0     0.9474  0.8571     0.9 
       SEX   669.0    9.0    8.0   677.0     0.9867  0.9882  0.9875 
  LOCATION  5784.0  149.0  206.0  5990.0     0.9749  0.9656  0.9702 
PROFESSION   249.0   17.0   27.0   276.0     0.9361  0.9022  0.9188 
       AGE   536.0   14.0   10.0   546.0     0.9745  0.9817  0.9781 
     macro       -      -      -       -          -       -  0.9636 
     macro       -      -      -       -          -       -  0.9736 
```