---
layout: model
title: Relation Extraction Between Dates and Clinical Entities (ReDL)
author: John Snow Labs
name: redl_date_clinical_biobert
date: 2023-01-14
tags: [licensed, en, clinical, relation_extraction, tensorflow]
task: Relation Extraction
language: en
nav_key: models
edition: Healthcare NLP 4.2.4
spark_version: 3.0
supported: true
engine: tensorflow
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Identify if tests were conducted on a particular date or any diagnosis was made on a specific date by checking relations between clinical entities and dates. 1 : Shows date and the clinical entity are related, 0 : Shows date and the clinical entity are not related.

## Predicted Entities

`1`, `0`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/RE_CLINICAL_DATE/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/10.1.Clinical_Relation_Extraction_BodyParts_Models.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/redl_date_clinical_biobert_en_4.2.4_3.0_1673731277460.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/redl_date_clinical_biobert_en_4.2.4_3.0_1673731277460.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
documenter = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

sentencer = SentenceDetector()\
    .setInputCols(["document"])\
    .setOutputCol("sentences")

tokenizer = Tokenizer()\
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
val documenter = new DocumentAssembler() 
    .setInputCol("text") 
    .setOutputCol("document")

val sentencer = new SentenceDetector()
    .setInputCols("document")
    .setOutputCol("sentences")

val tokenizer = new Tokenizer()
    .setInputCols("sentences")
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

val events_re_ner_chunk_filter = new RENerChunksFilter() 
    .setInputCols(Array("ner_chunks", "dependencies"))
    .setOutputCol("re_ner_chunks")

val events_re_Model = RelationExtractionDLModel() 
    .pretrained("redl_date_clinical_biobert", "en", "clinical/models")
    .setPredictionThreshold(0.5)
    .setInputCols(Array("re_ner_chunks", "sentences")) 
    .setOutputCol("relations")

val pipeline = new Pipeline().setStages(Array(documenter,sentencer,tokenizer,words_embedder,pos_tagger,events_ner_tagger,ner_chunker,dependency_parser,events_re_ner_chunk_filter,events_re_Model))

val data = Seq("This 73 y/o patient had CT on 1/12/95, with progressive memory and cognitive decline since 8/11/94.").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

## Results

```bash
+--------+-------+-------------+-----------+--------------------+-------+-------------+-----------+--------------------+----------+
|relation|entity1|entity1_begin|entity1_end|              chunk1|entity2|entity2_begin|entity2_end|              chunk2|confidence|
+--------+-------+-------------+-----------+--------------------+-------+-------------+-----------+--------------------+----------+
|       1|   TEST|           24|         25|                  CT|   DATE|           30|         36|             1/12/95|0.99997973|
|       1|   TEST|           24|         25|                  CT|PROBLEM|           44|         83|progressive memor...| 0.9998983|
|       1|   TEST|           24|         25|                  CT|   DATE|           91|         97|             8/11/94| 0.9997316|
|       1|   DATE|           30|         36|             1/12/95|PROBLEM|           44|         83|progressive memor...| 0.9998915|
|       1|   DATE|           30|         36|             1/12/95|   DATE|           91|         97|             8/11/94| 0.9997931|
|       1|PROBLEM|           44|         83|progressive memor...|   DATE|           91|         97|             8/11/94| 0.9998667|
+--------+-------+-------------+-----------+--------------------+-------+-------------+-----------+--------------------+----------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|redl_date_clinical_biobert|
|Compatibility:|Healthcare NLP 4.2.4+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|
|Size:|401.7 MB|

## References

Trained on an internal dataset.

## Benchmarking

```bash
label              Recall Precision        F1   Support
0                   0.738     0.729     0.734        84
1                   0.945     0.947     0.946       416
Avg.                0.841     0.838     0.840        -
```
