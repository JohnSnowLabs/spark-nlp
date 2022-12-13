---
layout: model
title: Legal Relation Extraction (Confidentiality, md, Unidirectional)
author: John Snow Labs
name: legre_confidentiality_md
date: 2022-11-09
tags: [en, legal, licensed, confidentiality, re]
task: Relation Extraction
language: en
edition: Legal NLP 1.0.0
spark_version: 3.0
supported: true
annotator: RelationExtractionDLModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description
IMPORTANT: Don't run this model on the whole legal agreement. Instead:
- Split by paragraphs. You can use [notebook 1](https://github.com/JohnSnowLabs/spark-nlp-workshop/tree/master/tutorials/Certification_Trainings_JSL) in Finance or Legal as inspiration;
- Use the `legclf_cuad_confidentiality_clause` Text Classifier to select only these paragraphs; 

This is a Legal Relation Extraction Model to identify the Subject (who), Action (what), Object(the confidentiality) and Indirect Object (to whom) from confidentiality clauses. This model requires `legner_confidentiality` as an NER in the pipeline. It's a `md` model with Unidirectional Relations, meaning that the model retrieves in chunk1 the left side of the relation (source), and in chunk2 the right side (target).

## Predicted Entities

`is_confidentiality_indobject`, `is_confidentiality_object`, `is_confidentiality_subject`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/legal/models/legre_confidentiality_md_en_1.0.0_3.0_1668006317769.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/legal/models/legre_confidentiality_md_en_1.0.0_3.0_1668006317769.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

documentAssembler = nlp.DocumentAssembler()\
  .setInputCol("text")\
  .setOutputCol("document")

sentencizer = nlp.SentenceDetectorDLModel\
        .pretrained("sentence_detector_dl", "en") \
        .setInputCols(["document"])\
        .setOutputCol("sentence")

tokenizer = nlp.Tokenizer()\
        .setInputCols("sentence")\
        .setOutputCol("token")

pos_tagger = nlp.PerceptronModel()\
    .pretrained() \
    .setInputCols(["sentence", "token"])\
    .setOutputCol("pos_tags")

dependency_parser = nlp.DependencyParserModel() \
    .pretrained("dependency_conllu", "en") \
    .setInputCols(["sentence", "pos_tags", "token"]) \
    .setOutputCol("dependencies")

embeddings = nlp.RoBertaEmbeddings.pretrained("roberta_embeddings_legal_roberta_base","en") \
    .setInputCols(["sentence", "token"]) \
    .setOutputCol("embeddings")

ner_model = legal.NerModel.pretrained('legner_confidentiality', 'en', 'legal/models') \
        .setInputCols(["sentence", "token", "embeddings"]) \
        .setOutputCol("ner")

ner_converter = nlp.NerConverter() \
        .setInputCols(["sentence","token","ner"]) \
        .setOutputCol("ner_chunk")

re_filter = legal.RENerChunksFilter()\
    .setInputCols(["ner_chunk", "dependencies"])\
    .setOutputCol("re_ner_chunks")\
    .setMaxSyntacticDistance(10)\
    .setRelationPairs(['CONFIDENTIALITY_ACTION-CONFIDENTIALITY_SUBJECT','CONFIDENTIALITY_ACTION-CONFIDENTIALITY','CONFIDENTIALITY_SUBJECT-CONFIDENTIALITY_INDIRECT_OBJECT'])

reDL = legal.RelationExtractionDLModel.pretrained("legre_confidentiality_md", "en", "legal/models") \
    .setPredictionThreshold(0.5) \
    .setInputCols(["re_ner_chunks", "sentence"]) \
    .setOutputCol("relations")
    
pipeline = Pipeline(stages=[documentAssembler,sentencizer, tokenizer,pos_tagger,dependency_parser, embeddings, ner_model, ner_converter,re_filter, reDL])

text = """Each party acknowledges that the other's Confidential Information contains valuable trade secret  and proprietary information of that party."""

data = spark.createDataFrame([[text]]).toDF("text")
model = pipeline.fit(data)
res = model.transform(data)
```

</div>

## Results

```bash
+--------------------------+----------------------+-------------+-----------+------------+-----------------------+-------------+-----------+-----------------------------------------+----------+
|relation                  |entity1               |entity1_begin|entity1_end|chunk1      |entity2                |entity2_begin|entity2_end|chunk2                                   |confidence|
+--------------------------+----------------------+-------------+-----------+------------+-----------------------+-------------+-----------+-----------------------------------------+----------+
|is_confidentiality_subject|CONFIDENTIALITY_ACTION|11           |22         |acknowledges|CONFIDENTIALITY_SUBJECT|0            |9          |Each party                               |0.67629266|
|is_confidentiality_object |CONFIDENTIALITY_ACTION|11           |22         |acknowledges|CONFIDENTIALITY        |41           |64         |Confidential Information                 |0.99151576|
|is_confidentiality_object |CONFIDENTIALITY_ACTION|11           |22         |acknowledges|CONFIDENTIALITY        |84           |124        |trade secret  and proprietary information|0.98372066|
+--------------------------+----------------------+-------------+-----------+------------+-----------------------+-------------+-----------+-----------------------------------------+----------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|legre_confidentiality_md|
|Compatibility:|Legal NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|
|Size:|402.3 MB|

## References

Manual annotations on CUAD dataset

## Benchmarking

```bash
Relation                            Recall    Precision  F1          Support 

is_confidentiality_indobject         0.960     1.000     0.980        25 
is_confidentiality_object            1.000     0.933     0.966        56
is_confidentiality_subject           0.935     1.000     0.967        31
other                                0.989     1.000     0.994        88

Avg.                                 0.971     0.983     0.977

Weighted Avg.                        0.980     0.981     0.980
```
