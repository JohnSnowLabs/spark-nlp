---
layout: model
title: Extract relations between phenotypic abnormalities and diseases (ReDL)
author: John Snow Labs
name: redl_human_phenotype_gene_biobert
date: 2023-01-14
tags: [relation_extraction, en, licensed, clinical, tensorflow]
task: Relation Extraction
language: en
edition: Healthcare NLP 4.2.4
spark_version: 3.0
supported: true
engine: tensorflow
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Extract relations to fully understand the origin of some phenotypic abnormalities and their associated diseases. 1 : Entities are related, 0 : Entities are not related.

## Predicted Entities

`1`, `0`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/10.Clinical_Relation_Extraction.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/redl_human_phenotype_gene_biobert_en_4.2.4_3.0_1673737099610.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/redl_human_phenotype_gene_biobert_en_4.2.4_3.0_1673737099610.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

pos_tagger = PerceptronModel()\
    .pretrained("pos_clinical", "en", "clinical/models") \
    .setInputCols(["sentences", "tokens"])\
    .setOutputCol("pos_tags")

words_embedder = WordEmbeddingsModel() \
    .pretrained("embeddings_clinical", "en", "clinical/models") \
    .setInputCols(["sentences", "tokens"]) \
    .setOutputCol("embeddings")

ner_tagger = MedicalNerModel.pretrained("ner_human_phenotype_gene_clinical", "en", "clinical/models")\
    .setInputCols("sentences", "tokens", "embeddings")\
    .setOutputCol("ner_tags") 

ner_converter = NerConverterInternal() \
    .setInputCols(["sentences", "tokens", "ner_tags"]) \
    .setOutputCol("ner_chunks")

dependency_parser = DependencyParserModel() \
    .pretrained("dependency_conllu", "en") \
    .setInputCols(["sentences", "pos_tags", "tokens"]) \
    .setOutputCol("dependencies")

#Set a filter on pairs of named entities which will be treated as relation candidates
re_ner_chunk_filter = RENerChunksFilter() \
    .setInputCols(["ner_chunks", "dependencies"])\
    .setMaxSyntacticDistance(10)\
    .setOutputCol("re_ner_chunks")

# The dataset this model is trained to is sentence-wise. 
# This model can also be trained on document-level relations - in which case, while predicting, use "document" instead of "sentence" as input.
re_model = RelationExtractionDLModel()\
    .pretrained('redl_human_phenotype_gene_biobert', 'en', "clinical/models") \
    .setPredictionThreshold(0.5)\
    .setInputCols(["re_ner_chunks", "sentences"]) \
    .setOutputCol("relations")

pipeline = Pipeline(stages=[documenter, sentencer, tokenizer, pos_tagger, words_embedder, ner_tagger, ner_converter, dependency_parser, re_ner_chunk_filter, re_model])

text = """She has a retinal degeneration, hearing loss and renal failure, short stature, Mutations in the SH3PXD2B gene coding for the Tks4 protein are responsible for the autosomal recessive."""

data = spark.createDataFrame([[text]]).toDF("text")

p_model = pipeline.fit(data)

result = p_model.transform(data)
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

val pos_tagger = PerceptronModel()
    .pretrained("pos_clinical", "en", "clinical/models") 
    .setInputCols(Array("sentences", "tokens"))
    .setOutputCol("pos_tags")

val words_embedder = WordEmbeddingsModel()
    .pretrained("embeddings_clinical", "en", "clinical/models")
    .setInputCols(Array("sentences", "tokens"))
    .setOutputCol("embeddings")

val ner_tagger = MedicalNerModel.pretrained("ner_human_phenotype_gene_clinical", "en", "clinical/models")
    .setInputCols(Array("sentences", "tokens", "embeddings"))
    .setOutputCol("ner_tags") 

val ner_converter = new NerConverterInternal()
    .setInputCols(Array("sentences", "tokens", "ner_tags"))
    .setOutputCol("ner_chunks")

val dependency_parser = DependencyParserModel()
    .pretrained("dependency_conllu", "en")
    .setInputCols(Array("sentences", "pos_tags", "tokens"))
    .setOutputCol("dependencies")

// Set a filter on pairs of named entities which will be treated as relation candidates
val re_ner_chunk_filter = new RENerChunksFilter()
    .setInputCols(Array("ner_chunks", "dependencies"))
    .setMaxSyntacticDistance(10)
    .setOutputCol("re_ner_chunks")

// The dataset this model is trained to is sentence-wise. 
// This model can also be trained on document-level relations - in which case, while predicting, use "document" instead of "sentence" as input.
val re_model = RelationExtractionDLModel()
    .pretrained("redl_human_phenotype_gene_biobert", "en", "clinical/models")
    .setPredictionThreshold(0.5)
    .setInputCols(Array("re_ner_chunks", "sentences"))
    .setOutputCol("relations")

val pipeline = new Pipeline().setStages(Array(documenter, sentencer, tokenizer, pos_tagger, words_embedder, ner_tagger, ner_converter, dependency_parser, re_ner_chunk_filter, re_model))

val data = Seq("""She has a retinal degeneration, hearing loss and renal failure, short stature, Mutations in the SH3PXD2B gene coding for the Tks4 protein are responsible for the autosomal recessive.""").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

## Results

```bash
+--------+-------+-------------+-----------+--------------------+-------+-------------+-----------+-------------------+----------+
|relation|entity1|entity1_begin|entity1_end|              chunk1|entity2|entity2_begin|entity2_end|             chunk2|confidence|
+--------+-------+-------------+-----------+--------------------+-------+-------------+-----------+-------------------+----------+
|       0|     HP|           10|         29|retinal degeneration|     HP|           32|         43|       hearing loss|0.92880034|
|       0|     HP|           10|         29|retinal degeneration|     HP|           49|         61|      renal failure|0.93935645|
|       0|     HP|           10|         29|retinal degeneration|     HP|           64|         76|      short stature|0.92370766|
|       1|     HP|           10|         29|retinal degeneration|   GENE|           96|        103|           SH3PXD2B|0.63739055|
|       1|     HP|           10|         29|retinal degeneration|     HP|          162|        180|autosomal recessive|0.58393383|
|       0|     HP|           32|         43|        hearing loss|     HP|           49|         61|      renal failure| 0.9543991|
|       0|     HP|           32|         43|        hearing loss|     HP|           64|         76|      short stature| 0.8060494|
|       1|     HP|           32|         43|        hearing loss|   GENE|           96|        103|           SH3PXD2B| 0.8507128|
|       1|     HP|           32|         43|        hearing loss|     HP|          162|        180|autosomal recessive|0.90283227|
|       0|     HP|           49|         61|       renal failure|     HP|           64|         76|      short stature|0.85388213|
|       1|     HP|           49|         61|       renal failure|   GENE|           96|        103|           SH3PXD2B|0.76057386|
|       1|     HP|           49|         61|       renal failure|     HP|          162|        180|autosomal recessive|0.85482293|
|       1|     HP|           64|         76|       short stature|   GENE|           96|        103|           SH3PXD2B| 0.8951201|
|       1|     HP|           64|         76|       short stature|     HP|          162|        180|autosomal recessive| 0.9018232|
|       1|   GENE|           96|        103|            SH3PXD2B|     HP|          162|        180|autosomal recessive|0.97185487|
+--------+-------+-------------+-----------+--------------------+-------+-------------+-----------+-------------------+----------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|redl_human_phenotype_gene_biobert|
|Compatibility:|Healthcare NLP 4.2.4+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|
|Size:|401.7 MB|

## References

Trained on a silver standard corpus of human phenotype and gene annotations and their relations.

## Benchmarking

```bash
label              Recall Precision        F1   Support
0                   0.922     0.908     0.915       129
1                   0.831     0.855     0.843        71
Avg.                0.877     0.882     0.879         -
```