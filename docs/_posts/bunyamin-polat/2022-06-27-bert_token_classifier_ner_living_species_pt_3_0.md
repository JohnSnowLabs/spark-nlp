---
layout: model
title: Detect Living Species
author: John Snow Labs
name: bert_token_classifier_ner_living_species
date: 2022-06-27
tags: [pt, ner, clinical, licensed, bertfortokenclassification]
task: Named Entity Recognition
language: pt
edition: Healthcare NLP 3.5.3
spark_version: 3.0
supported: true
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Extract living species from clinical texts in Portuguese which is critical to scientific disciplines like medicine, biology, ecology/biodiversity, nutrition and agriculture. This model is trained with the `BertForTokenClassification` method from the `transformers` library and imported into Spark NLP.

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
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/bert_token_classifier_ner_living_species_pt_3.5.3_3.0_1656319516041.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
document_assembler = DocumentAssembler()\
.setInputCol("text")\
.setOutputCol("document")\

sentence_detector = SentenceDetectorDLModel.pretrained("sentence_detector_dl", "xx")\
.setInputCols(["document"])\
.setOutputCol("sentence")

tokenizer = Tokenizer()\
.setInputCols(["sentence"])\
.setOutputCol("token")

ner_model = MedicalBertForTokenClassifier.pretrained("bert_token_classifier_ner_living_species", "pt", "clinical/models")\
.setInputCols(["sentence", "token"])\
.setOutputCol("ner")\
.setCaseSensitive(True)\
.setMaxSentenceLength(512)

ner_converter = NerConverter()\
.setInputCols(["sentence", "token", "ner"])\
.setOutputCol("ner_chunk")

pipeline = Pipeline(stages=[
document_assembler, 
sentence_detector,
tokenizer,
ner_model,
ner_converter   
])

data = spark.createDataFrame([["""Uma rapariga de 16 anos com um historial pessoal de asma apresentou ao departamento de dermatologia com lesões cutâneas assintomáticas que tinham estado presentes durante 2 meses. A paciente tinha sido tratada com creme corticosteróide devido a uma suspeita inicial de eczema atópico, apesar do qual apresentava um crescimento progressivo marcado das lesões. Tinha um gato doméstico que ela nunca tinha levado ao veterinário. O exame físico revelou placas em forma de anel com uma borda periférica activa na parte superior das costas e nos aspectos laterais do pescoço e da face. Cultura local obtida por raspagem de tapete isolado Trichophyton rubrum. Com base em dados clínicos e cultura, foi estabelecido o diagnóstico de tinea incognito."""]]).toDF("text")

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

val ner_model = MedicalBertForTokenClassifier.pretrained("bert_token_classifier_ner_living_species", "pt", "clinical/models")
.setInputCols(Array("sentence", "token"))
.setOutputCol("ner")
.setCaseSensitive(True)
.setMaxSentenceLength(512)

val ner_converter = new NerConverter()
.setInputCols(Array("sentence", "token", "ner"))
.setOutputCol("ner_chunk")

val pipeline = new Pipeline().setStages(Array(document_assembler, 
sentence_detector,
tokenizer,
ner_model,
ner_converter))

val data = Seq("""Uma rapariga de 16 anos com um historial pessoal de asma apresentou ao departamento de dermatologia com lesões cutâneas assintomáticas que tinham estado presentes durante 2 meses. A paciente tinha sido tratada com creme corticosteróide devido a uma suspeita inicial de eczema atópico, apesar do qual apresentava um crescimento progressivo marcado das lesões. Tinha um gato doméstico que ela nunca tinha levado ao veterinário. O exame físico revelou placas em forma de anel com uma borda periférica activa na parte superior das costas e nos aspectos laterais do pescoço e da face. Cultura local obtida por raspagem de tapete isolado Trichophyton rubrum. Com base em dados clínicos e cultura, foi estabelecido o diagnóstico de tinea incognito.""").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("pt.med_ner.living_species.token_bert").predict("""Uma rapariga de 16 anos com um historial pessoal de asma apresentou ao departamento de dermatologia com lesões cutâneas assintomáticas que tinham estado presentes durante 2 meses. A paciente tinha sido tratada com creme corticosteróide devido a uma suspeita inicial de eczema atópico, apesar do qual apresentava um crescimento progressivo marcado das lesões. Tinha um gato doméstico que ela nunca tinha levado ao veterinário. O exame físico revelou placas em forma de anel com uma borda periférica activa na parte superior das costas e nos aspectos laterais do pescoço e da face. Cultura local obtida por raspagem de tapete isolado Trichophyton rubrum. Com base em dados clínicos e cultura, foi estabelecido o diagnóstico de tinea incognito.""")
```

</div>

## Results

```bash
+-------------------+-------+
|ner_chunk          |label  |
+-------------------+-------+
|rapariga           |HUMAN  |
|pessoal            |HUMAN  |
|paciente           |HUMAN  |
|gato               |SPECIES|
|veterinário        |HUMAN  |
|Trichophyton rubrum|SPECIES|
+-------------------+-------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_token_classifier_ner_living_species|
|Compatibility:|Healthcare NLP 3.5.3+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[ner]|
|Language:|pt|
|Size:|665.4 MB|
|Case sensitive:|true|
|Max sentence length:|512|

## References

[https://temu.bsc.es/livingner/2022/05/03/multilingual-corpus/](https://temu.bsc.es/livingner/2022/05/03/multilingual-corpus/)

## Benchmarking

```bash
label         precision  recall  f1-score  support 
B-HUMAN       0.75       0.95    0.84      2832    
B-SPECIES     0.64       0.85    0.73      2810    
I-HUMAN       0.79       0.37    0.51      180     
I-SPECIES     0.63       0.71    0.67      1107    
micro-avg     0.69       0.86    0.76      6929    
macro-avg     0.70       0.72    0.69      6929    
weighted-avg  0.69       0.86    0.76      6929  
```
