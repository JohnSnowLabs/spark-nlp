---
layout: model
title: Extract temporal relations among clinical events (ReDL)
author: John Snow Labs
name: redl_temporal_events_biobert
date: 2021-07-24
tags: [relation_extraction, en, clinical, licensed]
task: Relation Extraction
language: en
edition: Healthcare NLP 3.0.3
spark_version: 2.4
supported: true
annotator: RelationExtractionDLModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---


## Description


Extract relations between clinical events in terms of time. If an event occurred before, after, or overlaps another event.


## Predicted Entities


`AFTER`, `BEFORE`, `OVERLAP`


{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/RE_CLINICAL_EVENTS/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/10.Clinical_Relation_Extraction.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/redl_temporal_events_biobert_en_3.0.3_2.4_1627121501681.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/redl_temporal_events_biobert_en_3.0.3_2.4_1627121501681.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}


## How to use






<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
...
documenter = DocumentAssembler()\
.setInputCol("text")\
.setOutputCol("document")

sentencer = SentenceDetector()\
.setInputCols(["document"])\
.setOutputCol("sentences")

tokenizer = sparknlp.annotators.Tokenizer()\
.setInputCols(["sentences"])\
.setOutputCol("tokens")

pos_tagger = PerceptronModel()\
.pretrained("pos_clinical", "en", "clinical/models") \
.setInputCols(["sentences", "tokens"])\
.setOutputCol("pos_tags")

words_embedder = WordEmbeddingsModel() \
.pretrained("embeddings_clinical", "en", "clinical/models") \
.setInputCols(["sentences", "tokens"]) \
.setOutputCol("embeddings")

ner_tagger = MedicalNerModel.pretrained("ner_events_clinical", "en", "clinical/models")\
.setInputCols("sentences", "tokens", "embeddings")\
.setOutputCol("ner_tags") 

ner_converter = NerConverter() \
.setInputCols(["sentences", "tokens", "ner_tags"]) \
.setOutputCol("ner_chunks")

dependency_parser = DependencyParserModel() \
.pretrained("dependency_conllu", "en") \
.setInputCols(["sentences", "pos_tags", "tokens"]) \
.setOutputCol("dependencies")

re_ner_chunk_filter = RENerChunksFilter() \
.setInputCols(["ner_chunks", "dependencies"])\
.setMaxSyntacticDistance(10)\
.setOutputCol("re_ner_chunks")

re_model = RelationExtractionDLModel()\
.pretrained("redl_temporal_events_biobert", "en", "clinical/models") \
.setPredictionThreshold(0.5)\
.setInputCols(["re_ner_chunks", "sentences"]) \
.setOutputCol("relations")

pipeline = Pipeline(stages=[documenter, sentencer, tokenizer, pos_tagger, words_embedder, ner_tagger, ner_converter, dependency_parser, re_ner_chunk_filter, re_model])

text = "She is diagnosed with cancer in 1991. Then she was admitted to Mayo Clinic in May 2000 and discharged in October 2001"

data = spark.createDataFrame([[text]]).toDF("text")

p_model = pipeline.fit(data)

result = p_model.transform(data)
```
```scala
...
val documenter = new DocumentAssembler() 
.setInputCol("text") 
.setOutputCol("document")

val sentencer = new SentenceDetector()
.setInputCols("document")
.setOutputCol("sentences")

val tokenizer = new Tokenizer()
.setInputCols("sentences")
.setOutputCol("tokens")

val pos_tagger = PerceptronModel()
.pretrained("pos_clinical", "en", "clinical/models") 
.setInputCols(Array("sentences", "tokens"))
.setOutputCol("pos_tags")

val words_embedder = WordEmbeddingsModel()
.pretrained("embeddings_clinical", "en", "clinical/models")
.setInputCols(Array("sentences", "tokens"))
.setOutputCol("embeddings")

val ner_tagger = MedicalNerModel.pretrained("ner_events_clinical", "en", "clinical/models")
.setInputCols(Array("sentences", "tokens", "embeddings"))
.setOutputCol("ner_tags") 

val ner_converter = new NerConverter()
.setInputCols(Array("sentences", "tokens", "ner_tags"))
.setOutputCol("ner_chunks")

val dependency_parser = DependencyParserModel()
.pretrained("dependency_conllu", "en")
.setInputCols(Array("sentences", "pos_tags", "tokens"))
.setOutputCol("dependencies")

// Set a filter on pairs of named entities which will be treated as relation candidates
val re_ner_chunk_filter = RENerChunksFilter()
.setInputCols(Array("ner_chunks", "dependencies"))
.setMaxSyntacticDistance(10)
.setOutputCol("re_ner_chunks")

// The dataset this model is trained to is sentence-wise. 
// This model can also be trained on document-level relations - in which case, while predicting, use "document" instead of "sentence" as input.
val re_model = RelationExtractionDLModel()
.pretrained("redl_temporal_events_biobert", "en", "clinical/models")
.setPredictionThreshold(0.5)
.setInputCols(Array("re_ner_chunks", "sentences"))
.setOutputCol("relations")

val pipeline = new Pipeline().setStages(Array(documenter, sentencer, tokenizer, pos_tagger, words_embedder, ner_tagger, ner_converter, dependency_parser, re_ner_chunk_filter, re_model))

val data = Seq("""She is diagnosed with cancer in 1991. Then she was admitted to Mayo Clinic in May 2000 and discharged in October 2001""").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("en.relation.temporal_events").predict("""She is diagnosed with cancer in 1991. Then she was admitted to Mayo Clinic in May 2000 and discharged in October 2001""")
```

</div>


## Results


```bash
+--------+-------------+-------------+-----------+-----------+-------------+-------------+-----------+------------+----------+
|relation|      entity1|entity1_begin|entity1_end|     chunk1|      entity2|entity2_begin|entity2_end|      chunk2|confidence|
+--------+-------------+-------------+-----------+-----------+-------------+-------------+-----------+------------+----------+
|  BEFORE|   OCCURRENCE|            7|         15|  diagnosed|      PROBLEM|           22|         27|      cancer|0.78168863|
| OVERLAP|      PROBLEM|           22|         27|     cancer|         DATE|           32|         35|        1991| 0.8492274|
|   AFTER|   OCCURRENCE|           51|         58|   admitted|CLINICAL_DEPT|           63|         73| Mayo Clinic|0.85629463|
|  BEFORE|   OCCURRENCE|           51|         58|   admitted|   OCCURRENCE|           91|        100|  discharged| 0.6843513|
| OVERLAP|CLINICAL_DEPT|           63|         73|Mayo Clinic|         DATE|           78|         85|    May 2000| 0.7844673|
|  BEFORE|CLINICAL_DEPT|           63|         73|Mayo Clinic|   OCCURRENCE|           91|        100|  discharged|0.60411876|
| OVERLAP|CLINICAL_DEPT|           63|         73|Mayo Clinic|         DATE|          105|        116|October 2001|  0.540761|
|  BEFORE|         DATE|           78|         85|   May 2000|   OCCURRENCE|           91|        100|  discharged| 0.6042761|
| OVERLAP|         DATE|           78|         85|   May 2000|         DATE|          105|        116|October 2001|0.64867175|
|  BEFORE|   OCCURRENCE|           91|        100| discharged|         DATE|          105|        116|October 2001| 0.5302478|
+--------+-------------+-------------+-----------+-----------+-------------+-------------+-----------+------------+----------+
```


{:.model-param}
## Model Information


{:.table-model}
|---|---|
|Model Name:|redl_temporal_events_biobert|
|Compatibility:|Healthcare NLP 3.0.3+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|
|Case sensitive:|true|


## Data Source


Trained on temporal clinical events benchmark dataset.


## Benchmarking


```bash
Relation           Recall Precision        F1   Support
AFTER               0.332     0.655     0.440      2123
BEFORE              0.868     0.908     0.887     13817
OVERLAP             0.887     0.733     0.802      7860
Avg.                0.695     0.765     0.710		-
```
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTExODgyMDYyMDddfQ==
-->