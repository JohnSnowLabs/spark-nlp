---
layout: model
title: Relation Extraction Between Dates and Clinical Entities (ReDL)
author: John Snow Labs
name: redl_date_clinical_biobert
date: 2021-06-01
tags: [licensed, en, clinical, relation_extraction]
task: Relation Extraction
language: en
edition: Healthcare NLP 3.0.3
spark_version: 3.0
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
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/redl_date_clinical_biobert_en_3.0.3_3.0_1622583984107.zip){:.button.button-orange.button-orange-trans.arr.button-icon}


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

words_embedder = WordEmbeddingsModel()\
.pretrained("embeddings_clinical", "en", "clinical/models")\
.setInputCols(["sentences", "tokens"])\
.setOutputCol("embeddings")

pos_tagger = PerceptronModel()\
.pretrained("pos_clinical", "en", "clinical/models") \
.setInputCols(["sentences", "tokens"])\
.setOutputCol("pos_tags")

events_ner_tagger = MedicalNerModel.pretrained("ner_events_clinical", "en", "clinical/models")\
.setInputCols("sentences", "tokens", "embeddings")\
.setOutputCol("ner_tags") 

ner_chunker = NerConverterInternal()\
.setInputCols(["sentences", "tokens", "ner_tags"])\
.setOutputCol("ner_chunks")

dependency_parser = DependencyParserModel() \
.pretrained("dependency_conllu", "en") \
.setInputCols(["sentences", "pos_tags", "tokens"]) \
.setOutputCol("dependencies")

events_re_ner_chunk_filter = RENerChunksFilter() \
.setInputCols(["ner_chunks", "dependencies"])\
.setOutputCol("re_ner_chunks")

events_re_Model = RelationExtractionDLModel() \
.pretrained('redl_date_clinical_biobert', "en", "clinical/models")\
.setPredictionThreshold(0.5)\
.setInputCols(["re_ner_chunks", "sentences"]) \
.setOutputCol("relations")

pipeline = Pipeline(stages=[
documenter,
sentencer,
tokenizer, 
words_embedder, 
pos_tagger, 
events_ner_tagger,
ner_chunker,
dependency_parser,
events_re_ner_chunk_filter,
events_re_Model])

data = spark.createDataFrame([['''This 73 y/o patient had CT on 1/12/95, with progressive memory and cognitive decline since 8/11/94.''']]).toDF("text")

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

val words_embedder = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")
.setInputCols(Array("sentence", "token"))
.setOutputCol("embeddings")

val pos_tagger = PerceptronModel()
.pretrained("pos_clinical", "en", "clinical/models") 
.setInputCols(Array("sentences", "tokens"))
.setOutputCol("pos_tags")

val events_ner_tagger = MedicalNerModel.pretrained("ner_events_clinical", "en", "clinical/models")
.setInputCols(Array("sentences", "tokens", "embeddings"))
.setOutputCol("ner_tags")  

val ner_chunker = new NerConverterInternal()
.setInputCols(Array("sentences", "tokens", "ner_tags"))
.setOutputCol("ner_chunks")

val dependency_parser = DependencyParserModel()
.pretrained("dependency_conllu", "en")
.setInputCols(Array("sentences", "pos_tags", "tokens"))
.setOutputCol("dependencies")

val events_re_ner_chunk_filter = RENerChunksFilter() 
.setInputCols(Array("ner_chunks", "dependencies"))
.setOutputCol("re_ner_chunks")

val events_re_Model = RelationExtractionDLModel() 
.pretrained('redl_date_clinical_biobert', "en", "clinical/models")
.setPredictionThreshold(0.5)
.setInputCols(Array("re_ner_chunks", "sentences")) 
.setOutputCol("relations")

val pipeline = new Pipeline().setStages(Array(documenter,sentencer,tokenizer,words_embedder,pos_tagger,events_ner_tagger,ner_chunker,dependency_parser,events_re_ner_chunk_filter,events_re_Model))

val data = Seq("This 73 y/o patient had CT on 1/12/95, with progressive memory and cognitive decline since 8/11/94.").toDS.toDF("text")

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
|Compatibility:|Healthcare NLP 3.0.3+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|
|Case sensitive:|true|


## Data Source


Trained on an internal dataset.


## Benchmarking


```bash
Relation           Recall Precision        F1   Support
0                   0.738     0.729     0.734        84
1                   0.945     0.947     0.946       416
Avg.                0.841     0.838     0.840        -
```
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTE0OTcwMjE1MzddfQ==
-->