---
layout: article
title: Pipelines
permalink: /docs/en/pipelines
key: docs-pipelines
modify_date: "2019-05-29"
---

## Pretrained Pipelines

### English

| Pipelines            | Name                   | Language                                                                                                                  |
| -------------------- | ---------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| Explain Document ML  | `explain_document_ml`  | [en](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/explain_document_ml_en_2.0.2_2.4_1556661821108.zip)  |
| Explain Document DL  | `explain_document_dl`  | [en](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/explain_document_dl_en_2.0.2_2.4_1556530585689.zip)  |
| Entity Recognizer DL | `entity_recognizer_dl` | [en](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/entity_recognizer_dl_en_2.0.0_2.4_1553230844671.zip) |

### French

| Pipelines               | Name                  | Language                                                                                                                 |
| ----------------------- | --------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| Explain Document Large  | `explain_document_lg` | [fr](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/explain_document_lg_fr_2.0.2_2.4_1559054673712.zip) |
| Explain Document Medium | `explain_document_md` | [fr](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/explain_document_md_fr_2.0.2_2.4_1559118515465.zip) |

#### French pipelines (explain_document) include

* Toenization (French UD style)

* Lemmatization

* Part of Speech (French UD GSD)

* Word Embeddings (`glove_840B_300` for large and `glove_6B_300` for medium)

* Named Entity Recongnition (French WikiNER)

* Entity chunking

## How to use

### Online

To use Spark NLP pretrained pipelines, you can call `PretrainedPipeline` with pipeline's name and its language (default is `en`):

```python
pipeline = PretrainedPipeline('explain_document_dl', lang='en')
```

Same in Scala

```scala
val pipeline = PretrainedPipeline("explain_document_dl", lang="en")
```

### Offline

If you have any trouble using online pipelines or models in your environment (maybe it's air-gapped), you can directly download them for `offline` use.

After downloading offline models/pipelines and extracting them, here is how you can use them iside your code (the path could be a shared storage like HDFS in a cluster):

```scala
val advancedPipeline = PipelineModel.load("/tmp/explain_document_dl_en_2.0.2_2.4_1556530585689/")
// To use the loaded Pipeline for prediction
advancedPipeline.transform(predictionDF)
```

## Demo

### French Pipeline

`explain_document_lg`

```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val frenchExplainDocumentPipeline = PretrainedPipeline("explain_document_lg", language="fr")

val frenchTestDF = spark.createDataFrame(Seq(
(0, """
Pour la deuxième année consécutive, le cinéma asiatique repart avec la récompense suprême à Cannes.
Après avoir consacré Une affaire de famille du japonais Hirokazu Kore-eda en 2018, le jury du Festival de Cannes a attribué la Palme d'or au Coréen Bong Joon-ho pour Parasite.

Contrairement à Quentin Tarantino, le cinéma français ne repart pas les mains vides de la compétition cannoise.
Le Montfermeillois Ladj Ly a remporté samedi soir le prix du jury pour son premier long-métrage Les Misérables, tandis que Céline Sciamma s'est vue décerner le prix du scénario pour Portrait de la jeune fille en feu. Le Grand Prix a quant à lui été remis à la Franco-Sénégalaise Mati Diop pour son film Atlantique.
"""),
(1, "Emmanuel Jean-Michel Frédéric Macron est le fils de Jean-Michel Macron, né en 1950, médecin, professeur de neurologie au CHU d'Amiens4 et responsable d'enseignement à la faculté de médecine de cette même ville5, et de Françoise Noguès, médecin conseil à la Sécurité sociale"),
(2, "Apple cherche a acheter une startup anglaise pour 1 milliard de dollard.")
)).toDF("id", "text").withColumn("id", monotonically_increasing_id())

val pipelineDF = frenchExplainDocumentPipeline.transform(frenchTestDF)

pipelineDF.select("entities.result").show(false)

+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|result                                                                                                                                                                                                                                                                                 |
+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|[Cannes, Une affaire de famille, Hirokazu Kore-eda, Festival de Cannes, Palme d'or, Coréen Bong Joon-ho, Parasite, Quentin Tarantino, Montfermeillois Ladj Ly, Les Misérables, Céline Sciamma, Portrait de la jeune fille en feu, Grand Prix, Franco-Sénégalaise Mati Diop, Atlantique]|
|[Emmanuel Jean-Michel Frédéric Macron, Jean-Michel Macron, CHU d'Amiens4, Françoise Noguès, Sécurité sociale]                                                                                                                                                                          |
|[Apple]                                                                                                                                                                                                                                                                                |
+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+

```
