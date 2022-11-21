---
layout: model
title: Legal Indemnification Relation Extraction (sm, Bidirectional)
author: John Snow Labs
name: legre_indemnifications
date: 2022-09-28
tags: [en, legal, re, indemnification, licensed]
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
- Use the `legclf_indemnification_clause` Text Classifier to select only these paragraphs; 

This is a Relation Extraction model to group the different entities extracted with the Indemnification NER model (see `legner_bert_indemnifications` in Models Hub). This model is a `sm` model without meaningful directions in the relations (the model was not trained to understand if the direction of the relation is from left to right or right to left).

There are bigger models in Models Hub trained also with directed relationships.

## Predicted Entities

`is_indemnification_subject`, `is_indemnification_object`, `is_indemnification_indobject`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/legal/LEGALRE_INDEMNIFICATION/){:.button.button-orange}
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/legal/models/legre_indemnifications_en_1.0.0_3.0_1664361611044.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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
        .setInputCols(["sentence"])\
        .setOutputCol("token")

tokenClassifier = legal.BertForTokenClassification.pretrained("legner_bert_indemnifications", "en", "legal/models")\
  .setInputCols("token", "sentence")\
  .setOutputCol("label")\
  .setCaseSensitive(True)

ner_converter = nlp.NerConverter()\
    .setInputCols(["sentence","token","label"])\
    .setOutputCol("ner_chunk")

# ONLY NEEDED IF YOU WANT TO FILTER RELATION PAIRS OR SYNTACTIC DISTANCE
# =================
pos_tagger = nlp.PerceptronModel()\
    .pretrained() \
    .setInputCols(["sentence", "token"])\
    .setOutputCol("pos_tags")

dependency_parser = nlp.DependencyParserModel() \
    .pretrained("dependency_conllu", "en") \
    .setInputCols(["sentence", "pos_tags", "token"]) \
    .setOutputCol("dependencies")

#Set a filter on pairs of named entities which will be treated as relation candidates
re_filter = legal.RENerChunksFilter()\
    .setInputCols(["ner_chunk", "dependencies"])\
    .setOutputCol("re_ner_chunks")\
    .setMaxSyntacticDistance(20)\
    .setRelationPairs(['INDEMNIFICATION_SUBJECT-INDEMNIFICATION_ACTION', 'INDEMNIFICATION_SUBJECT-INDEMNIFICATION_INDIRECT_OBJECT', 'INDEMNIFICATION_ACTION-INDEMNIFICATION', 'INDEMNIFICATION_ACTION-INDEMNIFICATION_INDIRECT_OBJECT'])
# =================

reDL = legal.RelationExtractionDLModel()\
    .pretrained("legre_indemnifications", "en", "legal/models")\
    .setPredictionThreshold(0.5)\
    .setInputCols(["re_ner_chunks", "sentence"])\
    .setOutputCol("relations")

nlpPipeline = Pipeline(stages=[
        documentAssembler,
        sentencizer,
        tokenizer,
        tokenClassifier,
        ner_converter,
        pos_tagger,
        dependency_parser,
        re_filter,
        reDL])

empty_data = spark.createDataFrame([[""]]).toDF("text")

model = nlpPipeline.fit(empty_data)

text='''The Company shall indemnify and hold harmless HOC against any losses, claims, damages or liabilities to which it may become subject under the 1933 Act or otherwise, insofar as such losses, claims, damages or liabilities (or actions in respect thereof) arise out of or are based upon '''

data = spark.createDataFrame([[text]]).toDF("text")
model = nlpPipeline.fit(data)
lmodel = LightPipeline(model)
res = lmodel.annotate(text)
```

</div>

## Results

```bash
relation	entity1	entity1_begin	entity1_end	chunk1	entity2	entity2_begin	entity2_end	chunk2	confidence
1	is_indemnification_subject	INDEMNIFICATION_SUBJECT	4	10	Company	INDEMNIFICATION_ACTION	32	44	hold harmless	0.8847967
2	is_indemnification_indobject	INDEMNIFICATION_SUBJECT	4	10	Company	INDEMNIFICATION_INDIRECT_OBJECT	46	48	HOC	0.96191925
3	is_indemnification_indobject	INDEMNIFICATION_ACTION	12	26	shall indemnify	INDEMNIFICATION_INDIRECT_OBJECT	46	48	HOC	0.7332646
10	is_indemnification_object	INDEMNIFICATION_ACTION	32	44	hold harmless	INDEMNIFICATION	70	75	claims	0.9728908
11	is_indemnification_object	INDEMNIFICATION_ACTION	32	44	hold harmless	INDEMNIFICATION	78	84	damages	0.9727499
12	is_indemnification_object	INDEMNIFICATION_ACTION	32	44	hold harmless	INDEMNIFICATION	89	99	liabilities	0.964168
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|legre_indemnifications|
|Compatibility:|Legal NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|
|Size:|405.9 MB|

## References

In-house annotated examples from CUAD legal dataset

## Benchmarking

```bash
                       label    Recall Precision        F1   Support
is_indemnification_indobject     0.966     1.000     0.982        29
is_indemnification_object        0.929     0.929     0.929        42
is_indemnification_subject       0.931     0.931     0.931        29
no_rel                           0.950     0.941     0.945       100
Avg.                             0.944     0.950     0.947        -
Weighted-Avg.                    0.945     0.945     0.945        -
```
