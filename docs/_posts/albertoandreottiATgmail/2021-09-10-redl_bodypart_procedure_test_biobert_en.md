---
layout: model
title: Relation Extraction Between Body Parts and Procedures
author: John Snow Labs
name: redl_bodypart_procedure_test_biobert
date: 2021-09-10
tags: [relation_extraction, en, clinical, dl, licensed]
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


Relation extraction between body parts entities like ‘Internal_organ_or_component’, ’External_body_part_or_region’ etc. and procedure and test entities. `1` : body part and test/procedure are related to each other.  `0` : body part and test/procedure are not related to each other.


## Predicted Entities


`0`, `1`


{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/redl_bodypart_procedure_test_biobert_en_3.0.3_2.4_1631307197287.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/redl_bodypart_procedure_test_biobert_en_3.0.3_2.4_1631307197287.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3} 


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

ner_tagger = MedicalNerModel.pretrained("ner_jsl_greedy", "en", "clinical/models")\
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
.setRelationPairs(["external_body_part_or_region-test"])

# The dataset this model is trained to is sentence-wise. 
# This model can also be trained on document-level relations - in which case, while predicting, use "document" instead of "sentence" as input.
re_model = RelationExtractionDLModel()\
.pretrained('redl_bodypart_procedure_test_biobert', 'en', "clinical/models") \
.setPredictionThreshold(0.5)\
.setInputCols(["re_ner_chunks", "sentences"]) \
.setOutputCol("relations")

pipeline = Pipeline(stages=[documenter, sentencer, tokenizer, pos_tagger, words_embedder, ner_tagger, ner_converter, dependency_parser, re_ner_chunk_filter, re_model])

data = spark.createDataFrame([['''TECHNIQUE IN DETAIL: After informed consent was obtained from the patient and his mother, the chest was scanned with portable ultrasound.''']]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
...
val documenter = new DocumentAssembler() 
.setInputCol("text") 
.setOutputCol("document")

val sentencer = new SentenceDetector()
.setInputCols(Array("document"))
.setOutputCol("sentences")

val tokenizer = new Tokenizer()
.setInputCols(Array("sentences"))
.setOutputCol("tokens")

val pos_tagger = PerceptronModel()
.pretrained("pos_clinical", "en", "clinical/models") 
.setInputCols(Array("sentences", "tokens"))
.setOutputCol("pos_tags")

val words_embedder = WordEmbeddingsModel()
.pretrained("embeddings_clinical", "en", "clinical/models")
.setInputCols(Array("sentences", "tokens"))
.setOutputCol("embeddings")

val ner_tagger = MedicalNerModel.pretrained("ner_jsl_greedy", "en", "clinical/models")
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
.setOutputCol("re_ner_chunks").
setRelationPairs(Array("external_body_part_or_region-test"))

// The dataset this model is trained to is sentence-wise. 
// This model can also be trained on document-level relations - in which case, while predicting, use "document" instead of "sentence" as input.
val re_model = RelationExtractionDLModel()
.pretrained("redl_bodypart_procedure_test_biobert", "en", "clinical/models")
.setPredictionThreshold(0.5)
.setInputCols(Array("re_ner_chunks", "sentences"))
.setOutputCol("relations")

val pipeline = new Pipeline().setStages(Array(documenter, sentencer, tokenizer, pos_tagger, words_embedder, ner_tagger, ner_converter, dependency_parser, re_ner_chunk_filter, re_model))

val data = Seq("""TECHNIQUE IN DETAIL: After informed consent was obtained from the patient and his mother, the chest was scanned with portable ultrasound.""").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("en.relation.bodypart.procedure").predict("""TECHNIQUE IN DETAIL: After informed consent was obtained from the patient and his mother, the chest was scanned with portable ultrasound.""")
```

</div>


## Results


```bash
|    |   relation | entity1                      | chunk1   | entity2   | chunk2              |   confidence |
|---:|-----------:|:-----------------------------|:---------|:----------|:--------------------|-------------:|
|  0 |          1 | External_body_part_or_region | chest    | Test      | portable ultrasound |      0.99953 |
```


{:.model-param}
## Model Information


{:.table-model}
|---|---|
|Model Name:|redl_bodypart_procedure_test_biobert_en_3.0.3_2.4|
|Compatibility:|Healthcare NLP 3.0.3+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|
|Case sensitive:|true|


## Data Source


Trained on a custom internal dataset.


## Benchmarking


```bash
Relation   Recall Precision        F1   Support
0           0.338     0.472     0.394       325
1           0.904     0.843     0.872      1275
Avg.        0.621     0.657     0.633         -
```
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTIwMTYwOTQyMjJdfQ==
-->