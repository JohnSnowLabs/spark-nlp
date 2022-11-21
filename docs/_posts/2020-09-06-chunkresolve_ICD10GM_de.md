---
layout: model
title: ICD10GM ChunkResolver
author: John Snow Labs
name: chunkresolve_ICD10GM
class: ChunkEntityResolverModel
language: de
repository: clinical/models
date: 2020-09-06
task: Entity Resolution
edition: Healthcare NLP 2.5.5
spark_version: 2.4
tags: [clinical,licensed,entity_resolution,de]
deprecated: true
annotator: ChunkEntityResolverModel
article_header:
   type: cover
use_language_switcher: "Python-Scala-Java"
---

{:.h2_title}
## Description
Entity Resolution model Based on KNN using Word Embeddings + Word Movers Distance.

## Predicted Entities
Codes and their normalized definition with `clinical_embeddings`.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/14.German_Healthcare_Models.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}{:target="_blank"}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/chunkresolve_ICD10GM_de_2.5.5_2.4_1599431635423.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

{:.h2_title}
## How to use
<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
...
icd10_resolution = ChunkEntityResolverModel.pretrained("chunkresolve_ICD10GM",'de','clinical/models') \
  .setInputCols(["token", "chunk_embeddings"]) \
  .setOutputCol("icd10_de_code")\
  .setDistanceFunction("EUCLIDEAN") \
  .setNeighbours(5)

pipeline_icd10 = Pipeline(stages = [documentAssembler, sentenceDetector, tokenizer, stopwords, de_embeddings, de_ner, ner_converter, chunk_embeddings, icd10_resolution])

empty_data = spark.createDataFrame([['''Das Kleinzellige Bronchialkarzinom (Kleinzelliger Lungenkrebs, SCLC) ist Hernia femoralis, Akne, einseitig, ein hochmalignes bronchogenes Karzinom, das überwiegend im Zentrum der Lunge, in einem Hauptbronchus entsteht. Die mittlere Prävalenz wird auf 1/20.000 geschätzt. Vom SCLC sind hauptsächlich Peronen mittleren Alters (27-66 Jahre) mit Raucheranamnese betroffen. Etwa 70% der Patienten mit SCLC haben bei Stellung der Diagnose schon extra-thorakale Symptome. Zu den Symptomen gehören Thoraxschmerz, Dyspnoe, Husten und pfeifende Atmung. Die Beteiligung benachbarter Bereiche verursacht Heiserkeit, Dysphagie und Oberes Vena-cava-Syndrom (Obstruktion des Blutflusses durch die Vena cava superior). Zusätzliche Symptome als Folge einer Fernmetastasierung sind ebenfalls möglich. Rauchen und Strahlenexposition sind synergistisch wirkende Risikofaktoren. Die industrielle Exposition mit Bis (Chlormethyläther) ist ein weiterer Risikofaktor. Röntgenaufnahmen des Thorax sind nicht ausreichend empfindlich, um einen SCLC frühzeitig zu erkennen. Röntgenologischen Auffälligkeiten muß weiter nachgegangen werden, meist mit Computertomographie. Die Diagnose wird bioptisch gesichert. Patienten mit SCLC erhalten meist Bestrahlung und/oder Chemotherapie. In Hinblick auf eine Verbesserung der Überlebenschancen der Patienten ist sowohl bei ausgedehnten und bei begrenzten SCLC eine kombinierte Chemotherapie wirksamer als die Behandlung mit Einzelsubstanzen. Es kann auch eine prophylaktische Bestrahlung des Schädels erwogen werden, da innerhalb von 2-3 Jahren nach Behandlungsbeginn ein hohes Risiko für zentralnervöse Metastasen besteht. Das Kleinzellige Bronchialkarzinom ist der aggressivste Lungentumor: Die 5-Jahres-Überlebensrate beträgt 1-5%, der Median des gesamten Überlebens liegt bei etwa 6 bis 10 Monaten.''']]).toDF("text")

model = pipeline_icd10.fit(empty_data)

results = model.transform(data)

```

```scala
...

val icd10_resolution = ChunkEntityResolverModel.pretrained("chunkresolve_ICD10GM",'de','clinical/models')
  .setInputCols("token", "chunk_embeddings")
  .setOutputCol("icd10_de_code")
  .setDistanceFunction("EUCLIDEAN")
  .setNeighbours(5)

val pipeline = new Pipeline().setStages(Array(documentAssembler, sentenceDetector, tokenizer, stopwords, de_embeddings, de_ner, ner_converter, chunk_embeddings, icd10_resolution))

val data = Seq("Das Kleinzellige Bronchialkarzinom (Kleinzelliger Lungenkrebs, SCLC) ist Hernia femoralis, Akne, einseitig, ein hochmalignes bronchogenes Karzinom, das überwiegend im Zentrum der Lunge, in einem Hauptbronchus entsteht. Die mittlere Prävalenz wird auf 1/20.000 geschätzt. Vom SCLC sind hauptsächlich Peronen mittleren Alters (27-66 Jahre) mit Raucheranamnese betroffen. Etwa 70% der Patienten mit SCLC haben bei Stellung der Diagnose schon extra-thorakale Symptome. Zu den Symptomen gehören Thoraxschmerz, Dyspnoe, Husten und pfeifende Atmung. Die Beteiligung benachbarter Bereiche verursacht Heiserkeit, Dysphagie und Oberes Vena-cava-Syndrom (Obstruktion des Blutflusses durch die Vena cava superior). Zusätzliche Symptome als Folge einer Fernmetastasierung sind ebenfalls möglich. Rauchen und Strahlenexposition sind synergistisch wirkende Risikofaktoren. Die industrielle Exposition mit Bis (Chlormethyläther) ist ein weiterer Risikofaktor. Röntgenaufnahmen des Thorax sind nicht ausreichend empfindlich, um einen SCLC frühzeitig zu erkennen. Röntgenologischen Auffälligkeiten muß weiter nachgegangen werden, meist mit Computertomographie. Die Diagnose wird bioptisch gesichert. Patienten mit SCLC erhalten meist Bestrahlung und/oder Chemotherapie. In Hinblick auf eine Verbesserung der Überlebenschancen der Patienten ist sowohl bei ausgedehnten und bei begrenzten SCLC eine kombinierte Chemotherapie wirksamer als die Behandlung mit Einzelsubstanzen. Es kann auch eine prophylaktische Bestrahlung des Schädels erwogen werden, da innerhalb von 2-3 Jahren nach Behandlungsbeginn ein hohes Risiko für zentralnervöse Metastasen besteht. Das Kleinzellige Bronchialkarzinom ist der aggressivste Lungentumor: Die 5-Jahres-Überlebensrate beträgt 1-5%, der Median des gesamten Überlebens liegt bei etwa 6 bis 10 Monaten.").toDF("text")
val result = pipeline.fit(data).transform(data)

```
</div>

{:.h2_title}
## Results

```bash
| Problem            | ICD10-Code |
|--------------------|------------|
| Kleinzellige       | M01.00     |
| Bronchialkarzinom  | I50.0      |
| Kleinzelliger      | I37.0      |
| Lungenkrebs        | B90.9      |
| SCLC               | C83.0      |
| ...                | ...        |
| Kleinzellige       | M01.00     |
| Bronchialkarzinom  | I50.0      |
| Lungentumor        | C90.31     |
| 1-5%               | I37.0      |
| 6 bis 10 Monaten   | Q91.6      |
```

{:.model-param}
## Model Information

{:.table-model}
|----------------|--------------------------|
| Name:           | chunkresolve_ICD10GM     |
| Type:    | ChunkEntityResolverModel |
| Compatibility:  | Spark NLP 2.5.5+                    |
| License:        | Licensed                 |
|Edition:|Official|               |
|Input labels:         | [token, chunk_embeddings]  |
|Output labels:        | [entity]                  |
| Language:       | de                       |
| Case sensitive: | True                     |
| Dependencies:  | w2v_cc_300d              |

{:.h2_title}
## Data Source
FILLUP
