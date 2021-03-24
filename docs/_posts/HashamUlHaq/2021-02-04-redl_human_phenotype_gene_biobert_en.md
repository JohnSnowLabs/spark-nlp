---
layout: model
title: Extract relations between phenotypic abnormalities and diseases (ReDL)
author: John Snow Labs
name: redl_human_phenotype_gene_biobert
date: 2021-02-04
task: Relation Extraction
language: en
edition: Spark NLP 2.7.3
tags: [licensed, clinical, en, relation_extraction]
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Extract relations to fully understand the origin of some phenotypic abnormalities and their associated diseases.

## Predicted Entities

`1` : Entities are related
`0` : Entities are not related

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/10.Clinical_Relation_Extraction.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/redl_human_phenotype_gene_biobert_en_2.7.3_2.4_1612440673031.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
...
words_embedder = WordEmbeddingsModel() \
    .pretrained("embeddings_clinical", "en", "clinical/models") \
    .setInputCols(["sentences", "tokens"]) \
    .setOutputCol("embeddings")
ner_tagger = NerDLModel() \
    .pretrained("ner_human_phenotype_gene_clinical", "en", "clinical/models") \
    .setInputCols(["sentences", "tokens", "embeddings"]) \
    .setOutputCol("ner_tags")
ner_converter = NerConverter() \
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
    .setOutputCol("re_ner_chunks")#.setRelationPairs(['SYMPTOM-EXTERNAL_BODY_PART_OR_REGION'])

# The dataset this model is trained to is sentence-wise. 
# This model can also be trained on document-level relations - in which case, while predicting, use "document" instead of "sentence" as input.
re_model = RelationExtractionDLModel()\
    .pretrained('redl_human_phenotype_gene_biobert', 'en', "clinical/models") \
    .setPredictionThreshold(0.5)\
    .setInputCols(["re_ner_chunks", "sentences"]) \
    .setOutputCol("relations")
pipeline = Pipeline(stages=[documenter, sentencer, tokenizer, pos_tagger, words_embedder, ner_tagger, ner_converter, dependency_parser, re_ner_chunk_filter, re_model])

text = "She has a retinal degeneration, hearing loss and renal failure, short stature, \
Mutations in the SH3PXD2B gene coding for the Tks4 protein are responsible for the autosomal recessive."

data = spark.createDataFrame([[text]]).toDF("text")
p_model = pipeline.fit(data)
result = p_model.transform(data)
```

```scala
...
val words_embedder = WordEmbeddingsModel()
    .pretrained("embeddings_clinical", "en", "clinical/models")
    .setInputCols(Array("sentences", "tokens"))
    .setOutputCol("embeddings")
val ner_tagger = NerDLModel()
    .pretrained("ner_clinical", "en", "clinical/models")
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
    .setOutputCol("re_ner_chunks").setRelationPairs(Array('SYMPTOM-EXTERNAL_BODY_PART_OR_REGION'))

// The dataset this model is trained to is sentence-wise. 
// This model can also be trained on document-level relations - in which case, while predicting, use "document" instead of "sentence" as input.
val re_model = RelationExtractionDLModel()
    .pretrained("redl_human_phenotype_gene_biobert", "en", "clinical/models")
    .setPredictionThreshold(0.5)
    .setInputCols(Array("re_ner_chunks", "sentences"))
    .setOutputCol("relations")
val pipeline = new Pipeline().setStages(Array(documenter, sentencer, tokenizer, pos_tagger, words_embedder, ner_tagger, ner_converter, dependency_parser, re_ner_chunk_filter, re_model))

val result = pipeline.fit(Seq.empty["She has a retinal degeneration, hearing loss and renal failure, short stature, \
Mutations in the SH3PXD2B gene coding for the Tks4 protein are responsible for the autosomal recessive."].toDS.toDF("text")).transform(data)
```

</div>

## Results

```bash
|    |   relation | entity1   |   entity1_begin |   entity1_end | chunk1               | entity2   |   entity2_begin |   entity2_end | chunk2              |   confidence |
|---:|-----------:|:----------|----------------:|--------------:|:---------------------|:----------|----------------:|--------------:|:--------------------|-------------:|
|  0 |          0 | HP        |              10 |            29 | retinal degeneration | HP        |              32 |            43 | hearing loss        |     0.893809 |
|  1 |          0 | HP        |              10 |            29 | retinal degeneration | HP        |              49 |            61 | renal failure       |     0.958486 |
|  2 |          1 | HP        |              10 |            29 | retinal degeneration | HP        |             162 |           180 | autosomal recessive |     0.65584  |
|  3 |          0 | HP        |              32 |            43 | hearing loss         | HP        |              64 |            76 | short stature       |     0.707055 |
|  4 |          1 | HP        |              32 |            43 | hearing loss         | GENE      |              96 |           103 | SH3PXD2B            |     0.640802 |

```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|redl_human_phenotype_gene_biobert|
|Compatibility:|Spark NLP 2.7.3+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|

## Data Source

Trained on a silver standard corpus of human phenotype and gene annotations and their relations.

## Benchmarking

```bash
Relation           Recall Precision        F1   Support
0                   0.922     0.908     0.915       129
1                   0.831     0.855     0.843        71
Avg.                0.877     0.882     0.879
```