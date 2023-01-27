---
layout: model
title: Detect Living Species(embeddings_scielo_300d)
author: John Snow Labs
name: ner_living_species_300
date: 2022-11-22
tags: [licensed, clinical, es, ner]
task: Named Entity Recognition
language: es
edition: Healthcare NLP 4.2.2
spark_version: 3.0
supported: true
annotator: MedicalNerModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Extract living species from clinical texts in Spanish which is critical to scientific disciplines like medicine, biology, ecology/biodiversity, nutrition and agriculture. This model is trained using `embeddings_scielo_300d` embeddings.

It is trained on the [LivingNER](https://temu.bsc.es/livingner/) corpus that is composed of clinical case reports extracted from miscellaneous medical specialties including COVID, oncology, infectious diseases, tropical medicine, urology, pediatrics, and others.

## Predicted Entities

`HUMAN`, `SPECIES`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/1.Clinical_Named_Entity_Recognition_Model.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_living_species_300_es_4.2.2_3.0_1669127690723.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/ner_living_species_300_es_4.2.2_3.0_1669127690723.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

embeddings = WordEmbeddingsModel.pretrained("embeddings_scielo_300d","es","clinical/models")\
    .setInputCols(["sentence", "token"])\
    .setOutputCol("embeddings")

ner_model = MedicalNerModel.pretrained("ner_living_species_300", "es","clinical/models")\
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

data = spark.createDataFrame([["""Lactante varón de dos años. Antecedentes familiares sin interés. Antecedentes personales: Embarazo, parto y periodo neonatal normal. En seguimiento por alergia a legumbres, diagnosticado con diez meses por reacción urticarial generalizada con lentejas y garbanzos, con dieta de exclusión a legumbres desde entonces. En ésta visita la madre describe episodios de eritema en zona maxilar derecha con afectación ocular ipsilateral que se resuelve en horas tras la administración de corticoides. Le ha ocurrido en 5-6 ocasiones, en relación con la ingesta de alimentos previamente tolerados. Exploración complementaria: Cacahuete, ac(ige)19.2 Ku.arb/l. Resultados: Ante la sospecha clínica de Síndrome de Frey, se tranquiliza a los padres, explicándoles la naturaleza del cuadro y se cita para revisión anual."""]]).toDF("text")

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

val embeddings = WordEmbeddingsModel.pretrained("embeddings_scielo_300d","es","clinical/models")
    .setInputCols(Array("sentence", "token"))
    .setOutputCol("embeddings")

val ner_model = MedicalNerModel.pretrained("ner_living_species_300", "es","clinical/models")
    .setInputCols(Array("sentence", "token", "embeddings"))
    .setOutputCol("ner")

val ner_converter = new NerConverter()
    .setInputCols(Array("sentence", "token", "ner"))
    .setOutputCol("ner_chunk")

val pipeline = new PipelineModel().setStages(Array(document_assembler, 
                                                  sentence_detector,
                                                  tokenizer,
                                                  embeddings,
                                                  ner_model,
                                                  ner_converter))

val data = Seq("""Lactante varón de dos años. Antecedentes familiares sin interés. Antecedentes personales: Embarazo, parto y periodo neonatal normal. En seguimiento por alergia a legumbres, diagnosticado con diez meses por reacción urticarial generalizada con lentejas y garbanzos, con dieta de exclusión a legumbres desde entonces. En ésta visita la madre describe episodios de eritema en zona maxilar derecha con afectación ocular ipsilateral que se resuelve en horas tras la administración de corticoides. Le ha ocurrido en 5-6 ocasiones, en relación con la ingesta de alimentos previamente tolerados. Exploración complementaria: Cacahuete, ac(ige)19.2 Ku.arb/l. Resultados: Ante la sospecha clínica de Síndrome de Frey, se tranquiliza a los padres, explicándoles la naturaleza del cuadro y se cita para revisión anual.""").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```

{:.nlu-block}
```python
import nlu
nlu.load("es.med_ner.living_species.300").predict("""Lactante varón de dos años. Antecedentes familiares sin interés. Antecedentes personales: Embarazo, parto y periodo neonatal normal. En seguimiento por alergia a legumbres, diagnosticado con diez meses por reacción urticarial generalizada con lentejas y garbanzos, con dieta de exclusión a legumbres desde entonces. En ésta visita la madre describe episodios de eritema en zona maxilar derecha con afectación ocular ipsilateral que se resuelve en horas tras la administración de corticoides. Le ha ocurrido en 5-6 ocasiones, en relación con la ingesta de alimentos previamente tolerados. Exploración complementaria: Cacahuete, ac(ige)19.2 Ku.arb/l. Resultados: Ante la sospecha clínica de Síndrome de Frey, se tranquiliza a los padres, explicándoles la naturaleza del cuadro y se cita para revisión anual.""")
```
</div>

## Results

```bash
+--------------+-------+
|ner_chunk     |label  |
+--------------+-------+
|Lactante varón|HUMAN  |
|familiares    |HUMAN  |
|personales    |HUMAN  |
|neonatal      |HUMAN  |
|legumbres     |SPECIES|
|lentejas      |SPECIES|
|garbanzos     |SPECIES|
|legumbres     |SPECIES|
|madre         |HUMAN  |
|Cacahuete     |SPECIES|
|padres        |HUMAN  |
+--------------+-------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_living_species_300|
|Compatibility:|Healthcare NLP 4.2.2+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|es|
|Size:|15.0 MB|

## Benchmarking

```bash
label         precision  recall  f1-score  support 
B-HUMAN       0.98       0.97    0.98      3281    
B-SPECIES     0.94       0.98    0.96      3712    
I-HUMAN       0.87       0.81    0.84      297     
I-SPECIES     0.79       0.89    0.84      1732    
micro-avg     0.92       0.95    0.94      9022    
macro-avg     0.90       0.91    0.90      9022    
weighted-avg  0.93       0.95    0.94      9022 
```