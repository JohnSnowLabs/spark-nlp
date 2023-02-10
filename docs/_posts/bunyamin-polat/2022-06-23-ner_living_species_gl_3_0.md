---
layout: model
title: Detect Living Species (w2v_cc_300d)
author: John Snow Labs
name: ner_living_species
date: 2022-06-23
tags: [gl, ner, clinical, licensed]
task: Named Entity Recognition
language: gl
edition: Healthcare NLP 3.5.3
spark_version: 3.0
supported: true
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Extract living species from clinical texts in Galician which is critical to scientific disciplines like medicine, biology, ecology/biodiversity, nutrition and agriculture. This model is trained using `w2v_cc_300d` embeddings.

It is trained on the [LivingNER](https://temu.bsc.es/livingner/2022/05/03/multilingual-corpus/) corpus that is composed of clinical case reports extracted from miscellaneous medical specialties including COVID, oncology, infectious diseases, tropical medicine, urology, pediatrics, and others.

**NOTE :**
1.	The text files were translated from Spanish with a neural machine translation system.
2.	The annotations were translated with the same neural machine translation system.
3.	The translated annotations were transferred to the translated text files using an annotation transfer technology.

## Predicted Entities

`HUMAN`, `SPECIES`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/1.Clinical_Named_Entity_Recognition_Model.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_living_species_gl_3.5.3_3.0_1655976346794.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/ner_living_species_gl_3.5.3_3.0_1655976346794.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
document_assembler = DocumentAssembler()\
.setInputCol("text")\
.setOutputCol("document")

sentence_detector = SentenceDetectorDLModel.pretrained("sentence_detector_dl", "xx")\
.setInputCols(["document"])\
.setOutputCol("sentence")

tokenizer = Tokenizer() \
.setInputCols(["sentence"])\
.setOutputCol("token")

embeddings = WordEmbeddingsModel.pretrained("w2v_cc_300d", "gl")\
.setInputCols(["sentence", "token"])\
.setOutputCol("embeddings")

ner_model = MedicalNerModel.pretrained("ner_living_species", "gl", "clinical/models")\
.setInputCols(["sentence", "token", "embeddings"])\
.setOutputCol("ner")

ner_converter = NerConverter()\
.setInputCols(["sentence", "token", "ner"])\
.setOutputCol("ner_chunk")

pipeline = Pipeline(stages=[
document_assembler, 
sentence_detector,
tokenizer,
embeddings,
ner_model,
ner_converter   
])

data = spark.createDataFrame([["""Muller de 45 anos, sen antecedentes médicos de interese, que foi remitida á consulta de dermatoloxía de urxencias por lesións faciales de tres semanas de evolución. A paciente non presentaba lesións noutras localizaciones nin outra clínica de interese. No seu centro de saúde prescribíronlle corticoides tópicos ante a sospeita de picaduras de artrópodos e unha semana despois, antivirales orais baixo o diagnóstico de posible infección herpética. As lesións interferían de forma notable na súa vida persoal e profesional xa que traballaba de face ao púbico. Unha semana máis tarde o diagnóstico foi confirmado ao resultar o cultivo positivo a Staphylococcus aureus."""]]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala

val document_assembler = new DocumentAssembler()
.setInputCol("text")
.setOutputCol("document")

val sentence_detector = SentenceDetectorDLModel.pretrained("sentence_detector_dl", "xx")
.setInputCols(Array("document"))
.setOutputCol("sentence")

val tokenizer = new Tokenizer()
.setInputCols(Array("sentence"))
.setOutputCol("token")

val embeddings = WordEmbeddingsModel.pretrained("w2v_cc_300d","gl")
.setInputCols(Array("sentence", "token"))
.setOutputCol("embeddings")

val ner_model = MedicalNerModel.pretrained("ner_living_species", "gl","clinical/models")
.setInputCols(Array("sentence", "token", "embeddings"))
.setOutputCol("ner")

val ner_converter = new NerConverter()
.setInputCols(Array("sentence", "token", "ner"))
.setOutputCol("ner_chunk")

val pipeline = new Pipeline().setStages(Array(document_assembler, 
sentence_detector,
tokenizer,
embeddings,
ner_model,
ner_converter))

val data = Seq("""Muller de 45 anos, sen antecedentes médicos de interese, que foi remitida á consulta de dermatoloxía de urxencias por lesións faciales de tres semanas de evolución. A paciente non presentaba lesións noutras localizaciones nin outra clínica de interese. No seu centro de saúde prescribíronlle corticoides tópicos ante a sospeita de picaduras de artrópodos e unha semana despois, antivirales orais baixo o diagnóstico de posible infección herpética. As lesións interferían de forma notable na súa vida persoal e profesional xa que traballaba de face ao púbico. Unha semana máis tarde o diagnóstico foi confirmado ao resultar o cultivo positivo a Staphylococcus aureus.""").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("gl.med_ner.living_species").predict("""Muller de 45 anos, sen antecedentes médicos de interese, que foi remitida á consulta de dermatoloxía de urxencias por lesións faciales de tres semanas de evolución. A paciente non presentaba lesións noutras localizaciones nin outra clínica de interese. No seu centro de saúde prescribíronlle corticoides tópicos ante a sospeita de picaduras de artrópodos e unha semana despois, antivirales orais baixo o diagnóstico de posible infección herpética. As lesións interferían de forma notable na súa vida persoal e profesional xa que traballaba de face ao púbico. Unha semana máis tarde o diagnóstico foi confirmado ao resultar o cultivo positivo a Staphylococcus aureus.""")
```

</div>

## Results

```bash
+---------------------+-------+
|ner_chunk            |label  |
+---------------------+-------+
|Muller               |HUMAN  |
|paciente             |HUMAN  |
|artrópodos           |SPECIES|
|antivirales          |SPECIES|
|herpética            |SPECIES|
|púbico               |HUMAN  |
|Staphylococcus aureus|SPECIES|
+---------------------+-------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_living_species|
|Compatibility:|Healthcare NLP 3.5.3+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|gl|
|Size:|15.2 MB|

## References

[https://temu.bsc.es/livingner/2022/05/03/multilingual-corpus/](https://temu.bsc.es/livingner/2022/05/03/multilingual-corpus/)

## Benchmarking

```bash
label         precision  recall  f1-score  support 
B-HUMAN       0.88       0.97    0.92      2952    
B-SPECIES     0.54       0.91    0.67      3333    
I-HUMAN       0.74       0.75    0.74      206     
I-SPECIES     0.59       0.85    0.70      1297    
micro-avg     0.65       0.92    0.76      7788    
macro-avg     0.69       0.87    0.76      7788    
weighted-avg  0.68       0.92    0.77      7788  
```
