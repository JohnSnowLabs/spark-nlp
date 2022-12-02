---
layout: model
title: Detect Living Species (w2v_cc_300d)
author: John Snow Labs
name: ner_living_species
date: 2022-06-23
tags: [ca, ner, clinical, licensed]
task: Named Entity Recognition
language: ca
edition: Healthcare NLP 3.5.3
spark_version: 3.0
supported: true
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Extract living species from clinical texts in Catalan which is critical to scientific disciplines like medicine, biology, ecology/biodiversity, nutrition and agriculture. This model is trained using `w2v_cc_300d` embeddings.

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
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_living_species_ca_3.5.3_3.0_1655975861739.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

tokenizer = Tokenizer()\
.setInputCols(["sentence"])\
.setOutputCol("token")

embeddings = WordEmbeddingsModel.pretrained("w2v_cc_300d", "ca")\
.setInputCols(["sentence", "token"]) \
.setOutputCol("embeddings")

ner_model = MedicalNerModel.pretrained("ner_living_species", "ca", "clinical/models")\
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

data = spark.createDataFrame([["""Dona de 47 anys al·lèrgica al iode, fumadora social, intervinguda de varices, dues cesàries i un abscés gluti. Sense altres antecedents mèdics d'interès ni tractament habitual. Viu amb el seu marit i tres fills, treballa com a professora. En el moment de la nostra valoració en la planta de Cirurgia General, la pacient presenta TA 69/40 mm Hg, freqüència cardíaca 120 lpm, taquipnea en repòs, pal·lidesa mucocutánea, mala perfusió distal i afligeix nàusees. L'abdomen és tou, no presenta peritonismo i el dèbit del drenatge abdominal roman sense canvis. Les serologies de Coxiella burnetii, Bartonella henselae, Borrelia burgdorferi, Entamoeba histolytica, Toxoplasma gondii, citomegalovirus, virus de Epstein Barr, virus varicel·la zoster i parvovirus B19 van ser negatives. No obstant això, es va detectar test de rosa de Bengala positiu per a Brucella, el test de Coombs i les aglutinacions també van ser positives amb un títol 1/40."""]]).toDF("text")

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

val embeddings = WordEmbeddingsModel.pretrained("w2v_cc_300d", "ca")
.setInputCols(Array("sentence", "token"))
.setOutputCol("embeddings")

val ner_model = MedicalNerModel.pretrained("ner_living_species", "ca", "clinical/models")
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

val data = Seq("""Dona de 47 anys al·lèrgica al iode, fumadora social, intervinguda de varices, dues cesàries i un abscés gluti. Sense altres antecedents mèdics d'interès ni tractament habitual. Viu amb el seu marit i tres fills, treballa com a professora. En el moment de la nostra valoració en la planta de Cirurgia General, la pacient presenta TA 69/40 mm Hg, freqüència cardíaca 120 lpm, taquipnea en repòs, pal·lidesa mucocutánea, mala perfusió distal i afligeix nàusees. L'abdomen és tou, no presenta peritonismo i el dèbit del drenatge abdominal roman sense canvis. Les serologies de Coxiella burnetii, Bartonella henselae, Borrelia burgdorferi, Entamoeba histolytica, Toxoplasma gondii, citomegalovirus, virus de Epstein Barr, virus varicel·la zoster i parvovirus B19 van ser negatives. No obstant això, es va detectar test de rosa de Bengala positiu per a Brucella, el test de Coombs i les aglutinacions també van ser positives amb un títol 1/40.""").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("ca.med_ner.living_species").predict("""Dona de 47 anys al·lèrgica al iode, fumadora social, intervinguda de varices, dues cesàries i un abscés gluti. Sense altres antecedents mèdics d'interès ni tractament habitual. Viu amb el seu marit i tres fills, treballa com a professora. En el moment de la nostra valoració en la planta de Cirurgia General, la pacient presenta TA 69/40 mm Hg, freqüència cardíaca 120 lpm, taquipnea en repòs, pal·lidesa mucocutánea, mala perfusió distal i afligeix nàusees. L'abdomen és tou, no presenta peritonismo i el dèbit del drenatge abdominal roman sense canvis. Les serologies de Coxiella burnetii, Bartonella henselae, Borrelia burgdorferi, Entamoeba histolytica, Toxoplasma gondii, citomegalovirus, virus de Epstein Barr, virus varicel·la zoster i parvovirus B19 van ser negatives. No obstant això, es va detectar test de rosa de Bengala positiu per a Brucella, el test de Coombs i les aglutinacions també van ser positives amb un títol 1/40.""")
```

</div>

## Results

```bash
+-----------------------+-------+
|ner_chunk              |label  |
+-----------------------+-------+
|Dona                   |HUMAN  |
|marit                  |HUMAN  |
|fills                  |HUMAN  |
|professora             |HUMAN  |
|pacient                |HUMAN  |
|Coxiella burnetii      |SPECIES|
|Bartonella henselae    |SPECIES|
|Borrelia burgdorferi   |SPECIES|
|Entamoeba histolytica  |SPECIES|
|Toxoplasma gondii      |SPECIES|
|citomegalovirus        |SPECIES|
|virus de Epstein Barr  |SPECIES|
|virus varicel·la zoster|SPECIES|
|parvovirus B19         |SPECIES|
|Brucella               |SPECIES|
+-----------------------+-------+
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
|Language:|ca|
|Size:|15.1 MB|

## References

[https://temu.bsc.es/livingner/2022/05/03/multilingual-corpus/](https://temu.bsc.es/livingner/2022/05/03/multilingual-corpus/)

## Benchmarking

```bash
label         precision  recall  f1-score  support 
B-HUMAN       0.88       0.97    0.92      3036    
B-SPECIES     0.68       0.94    0.78      3354    
I-HUMAN       0.87       0.64    0.74      195     
I-SPECIES     0.75       0.83    0.79      1329    
micro-avg     0.76       0.92    0.83      7914    
macro-avg     0.79       0.84    0.81      7914    
weighted-avg  0.77       0.92    0.84      7914  
```
