---
layout: model
title: Detect Temporal Relations for Clinical Events (Enriched)
author: John Snow Labs
name: re_temporal_events_enriched_clinical
date: 2020-09-28
task: Relation Extraction
language: en
edition: Healthcare NLP 2.6.0
spark_version: 2.4
tags: [re, en, clinical, licensed]
supported: true
article_header:
   type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description
This model can be used to identify temporal relationships among clinical events.
## Predicted Entities
`BEFORE`, `AFTER`, `SIMULTANEOUS`, `BEGUN_BY`, `ENDED_BY`, `DURING`, `BEFORE_OVERLAP`

{:.btn-box}

[Live Demo](https://demo.johnsnowlabs.com/healthcare/RE_CLINICAL_EVENTS/){:.button.button-orange.button-orange-trans.co.button-icon}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/10.Clinical_Relation_Extraction.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/re_temporal_events_enriched_clinical_en_2.5.5_2.4_1597775105767.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
## How to use

Use as part of an nlp pipeline with the following stages: DocumentAssembler, SentenceDetector, Tokenizer, PerceptronModel, DependencyParserModel, WordEmbeddingsModel, NerDLModel, NerConverter, RelationExtractionModel.

In the table below, `re_temporal_events_enriched_clinical` RE model, its labels, optimal NER model, and meaningful relation pairs are illustrated.

 |               RE MODEL               |                              RE MODEL LABES                             |      NER MODEL      |          RE PAIRS         |
 |:------------------------------------:|:-----------------------------------------------------------------------:|:-------------------:|:-------------------------:|
 | re_temporal_events_enriched_clinical | BEFORE, AFTER, SIMULTANEOUS, BEGUN_BY, ENDED_BY, DURING, BEFORE_OVERLAP | ner_events_clinical | [“No need to set pairs.”] |


<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
document_assembler = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

sentence_detector = SentenceDetector()\
    .setInputCols(["document"])\
    .setOutputCol("sentences")

tokenizer = sparknlp.annotators.Tokenizer()\
    .setInputCols(["sentences"])\
    .setOutputCol("tokens")

pos_tagger = PerceptronModel().pretrained("pos_clinical", "en", "clinical/models") \
    .setInputCols(["sentences", "tokens"])\
    .setOutputCol("pos_tags")

word_embeddings = WordEmbeddingsModel().pretrained("embeddings_clinical", "en", "clinical/models") \
    .setInputCols(["sentences", "tokens"]) \
    .setOutputCol("embeddings")

clinical_ner = MedicalNerModel.pretrained("ner_clinical", "en", "clinical/models")\
    .setInputCols("sentences", "tokens", "embeddings")\
    .setOutputCol("ner_tags")

ner_converter = NerConverter() \
    .setInputCols(["sentences", "tokens", "ner_tags"]) \
    .setOutputCol("ner_chunks")

dependency_parser = DependencyParserModel().pretrained("dependency_conllu", "en") \
    .setInputCols(["sentences", "pos_tags", "tokens"]) \
    .setOutputCol("dependencies")

clinical_re_Model = RelationExtractionModel()\
    .pretrained("re_temporal_events_enriched_clinical", "en", 'clinical/models')\
    .setInputCols(["embeddings", "pos_tags", "ner_chunks", "dependencies"])\
    .setOutputCol("relations")\
    .setMaxSyntacticDistance(4) #default: 0
    
nlp_pipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, pos_tagger, word_embeddings, clinical_ner, ner_converter,  dependency_parser, clinical_re_Model])

light_pipeline = LightPipeline(nlp_pipeline.fit(spark.createDataFrame([['']]).toDF("text")))

annotations = light_pipeline.fullAnnotate("""The patient is a 56-year-old right-handed female with longstanding intermittent right low back pain, who was involved in a motor vehicle accident in September of 2005. At that time, she did not notice any specific injury, but five days later, she started getting abnormal right low back pain.""")

```

```scala
val document_assembler = new DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document")

val sentence_detector = new SentenceDetector()
    .setInputCols("document")
    .setOutputCol("sentences")

val tokenizer = new Tokenizer()\
    .setInputCols(["sentences"])\
    .setOutputCol("tokens")

val pos_tagger = PerceptronModel().pretrained("pos_clinical", "en", "clinical/models")
    .setInputCols(Array("sentences", "tokens"))
    .setOutputCol("pos_tags")

val word_embeddings = WordEmbeddingsModel().pretrained("embeddings_clinical", "en", "clinical/models")
    .setInputCols(Array("sentences", "tokens"))
    .setOutputCol("embeddings")

val clinical_ner = MedicalNerModel.pretrained("ner_clinical", "en", "clinical/models")
    .setInputCols(Array("sentences", "tokens", "embeddings"))
    .setOutputCol("ner_tags")

val ner_converter = new NerConverter() 
    .setInputCols(Array("sentences", "tokens", "ner_tags"))
    .setOutputCol("ner_chunks")

val dependency_parser = DependencyParserModel().pretrained("dependency_conllu", "en")
    .setInputCols(Array("sentences", "pos_tags", "tokens"))
    .setOutputCol("dependencies")

val clinical_re_Model = RelationExtractionModel()
    .pretrained("re_temporal_events_enriched_clinical", "en", "clinical/models")
    .setInputCols("embeddings", "pos_tags", "ner_chunks", "dependencies")
    .setOutputCol("relations")
    .setMaxSyntacticDistance(4)

val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, tokenizer, pos_tagger, word_embeddings, clinical_ner, ner_converter,  dependency_parser, clinical_re_Model))

val data = Seq("""The patient is a 56-year-old right-handed female with longstanding intermittent right low back pain, who was involved in a motor vehicle accident in September of 2005. At that time, she did not notice any specific injury, but five days later, she started getting abnormal right low back pain.""").toDS().toDF("text")

val result = pipeline.fit(data).transform(data)


```
</div>

{:.h2_title}
## Results

```bash
+----+------------+-----------+-----------------+---------------+-----------------------------------------------+------------+-----------------+---------------+--------------------------+--------------+
|    | relation   | entity1   |   entity1_begin |   entity1_end | chunk1                                        | entity2    |   entity2_begin |   entity2_end | chunk2                   |   confidence |
+====+============+===========+=================+===============+===============================================+============+=================+===============+==========================+==============+
|  0 | OVERLAP    | PROBLEM   |              54 |            98 | longstanding intermittent right low back pain | OCCURRENCE |             121 |           144 | a motor vehicle accident |     0.532308 |
+----+------------+-----------+-----------------+---------------+-----------------------------------------------+------------+-----------------+---------------+--------------------------+--------------+
|  1 | AFTER      | DATE      |             171 |           179 | that time                                     | PROBLEM    |             201 |           219 | any specific injury      |     0.577288 |
+----+------------+-----------+-----------------+---------------+-----------------------------------------------+------------+-----------------+---------------+--------------------------+--------------+
```
{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|re_temporal_events_enriched_clinical|
|Type:|re|
|Compatibility:|Healthcare NLP 2.6.0 +|
|Edition:|Official|
|License:|Licensed|
|Input Labels:|[embeddings, pos_tags, ner_chunks, dependencies]|
|Output Labels:|[relations]|
|Language:|[en]|
|Case sensitive:|false|

{:.h2_title}
## Data Source
Trained on data gathered and manually annotated by John Snow Labs
https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/

{:.h2_title}
## Benchmarking
```bash
|Relation  | Recall  | Precision | F1   |
|---------:|--------:|----------:|-----:|
| OVERLAP  |  0.81   |  0.73     | 0.77 |
| BEFORE   |  0.85   |  0.88     | 0.86 |
| AFTER    |  0.38   |  0.46     | 0.43 |
```