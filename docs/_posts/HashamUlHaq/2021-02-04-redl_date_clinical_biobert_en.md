---
layout: model
title: Relation extraction between dates and clinical entities (ReDL)
author: John Snow Labs
name: redl_date_clinical_biobert
date: 2021-02-04
task: Relation Extraction
language: en
edition: Healthcare NLP 2.7.3
spark_version: 2.4
tags: [licensed, clinical, en, relation_extraction]
supported: true
annotator: RelationExtractionDLModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Identify if tests were conducted on a particular date or any diagnosis was made on a specific date by checking relations between clinical entities and dates. `1` : Shows date and the clinical entity are related, `0` : Shows date and the clinical entity are not related.

## Predicted Entities

`0`, `1`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/RE_CLINICAL_DATE/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/10.1.Clinical_Relation_Extraction_BodyParts_Models.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/redl_date_clinical_biobert_en_2.7.3_2.4_1612448249418.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

ner_tagger = MedicalNerModel.pretrained("jsl_ner_wip_greedy_clinical", "en", "clinical/models")\
.setInputCols("sentences", "tokens", "embeddings")\
.setOutputCol("ner_tags")

ner_converter = NerConverter() \
.setInputCols(["sentences", "tokens", "ner_tags"]) \
.setOutputCol("ner_chunks")

dependency_parser = DependencyParserModel() \
.pretrained("dependency_conllu", "en") \
.setInputCols(["sentences", "pos_tags", "tokens"]) \
.setOutputCol("dependencies")

# Set a filter on pairs of named entities which will be treated as relation candidates
re_ner_chunk_filter = RENerChunksFilter() \
.setInputCols(["ner_chunks", "dependencies"])\
.setMaxSyntacticDistance(10)\
.setOutputCol("re_ner_chunks")\
.setRelationPairs(['symptom-date', 'date-procedure', 'relativedate-test', 'test-date'])

# The dataset this model is trained to is sentence-wise. 
# This model can also be trained on document-level relations - in which case, while predicting, use "document" instead of "sentence" as input.
re_model = RelationExtractionDLModel()\
.pretrained('redl_date_clinical_biobert', 'en', "clinical/models") \
.setPredictionThreshold(0.5)\
.setInputCols(["re_ner_chunks", "sentences"]) \
.setOutputCol("relations")

pipeline = Pipeline(stages=[documenter, sentencer, tokenizer, pos_tagger, words_embedder, ner_tagger, ner_converter, dependency_parser, re_ner_chunk_filter, re_model])

text ="This 73 y/o patient had CT on 1/12/95, with progressive memory and cognitive decline since 8/11/94."
data = spark.createDataFrame([[text]]).toDF("text")
p_model = pipeline.fit(data)
result = p_model.transform(data)
```

```scala
...
val documenter = DocumentAssembler() 
.setInputCol("text") 
.setOutputCol("document")

val sentencer = SentenceDetector()
.setInputCols("document")
.setOutputCol("sentences")

val tokenizer = sparknlp.annotators.Tokenizer()
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

val ner_tagger = MedicalNerModel.pretrained("jsl_ner_wip_greedy_clinical", "en", "clinical/models")
.setInputCols(Array("sentences", "tokens", "embeddings"))
.setOutputCol("ner_tags") 

val ner_converter = NerConverter()
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
.setRelationPairs(Array('symptom-date', 'date-procedure', 'relativedate-test', 'test-date'))

// The dataset this model is trained to is sentence-wise. 
// This model can also be trained on document-level relations - in which case, while predicting, use "document" instead of "sentence" as input.
val re_model = RelationExtractionDLModel()
.pretrained("redl_date_clinical_biobert", "en", "clinical/models")
.setPredictionThreshold(0.5)
.setInputCols(Array("re_ner_chunks", "sentences"))
.setOutputCol("relations")

val pipeline = new Pipeline().setStages(Array(documenter, sentencer, tokenizer, pos_tagger, words_embedder, ner_tagger, ner_converter, dependency_parser, re_ner_chunk_filter, re_model))

val data = Seq("This 73 y/o patient had CT on 1/12/95, with progressive memory and cognitive decline since 8/11/94.").toDF("text")
val result = pipeline.fit(data).transform(data)
```



{:.nlu-block}
```python
import nlu
nlu.load("en.relation.date").predict("""This 73 y/o patient had CT on 1/12/95, with progressive memory and cognitive decline since 8/11/94.""")
```

</div>

## Results

```bash
|   | relations | entity1 | entity1_begin | entity1_end | chunk1                                   | entity2 | entity2_end | entity2_end | chunk2  | confidence |
|---|-----------|---------|---------------|-------------|------------------------------------------|---------|-------------|-------------|---------|------------|
| 0 | 1         | Test    | 24            | 25          | CT                                       | Date    | 31          | 37          | 1/12/95 | 1.0        |
| 1 | 1         | Symptom | 45            | 84          | progressive memory and cognitive decline | Date    | 92          | 98          | 8/11/94 | 1.0        |

```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|redl_date_clinical_biobert|
|Compatibility:|Healthcare NLP 2.7.3+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|

## Data Source

Trained on an internal dataset.

## Benchmarking

```bash
Relation           Recall Precision        F1   Support
0                   0.738     0.729     0.734        84
1                   0.945     0.947     0.946       416
Avg.                0.841     0.838     0.840
```