---
layout: model
title: Legal Relation Extraction (Indemnifications, md, Unidirectional)
author: John Snow Labs
name: legre_indemnifications_md
date: 2022-11-09
tags: [en, legal, licensed, indemnifications, re]
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

This is a Relation Extraction model to group the different entities extracted with the Indemnification NER model (see `legner_bert_indemnifications` in Models Hub). This model requires `legner_bert_indemnifications` as an NER in the pipeline. It's a `md` model with Unidirectional Relations, meaning that the model retrieves in chunk1 the left side of the relation (source), and in chunk2 the right side (target).

## Predicted Entities

`is_indemnification_subject`, `is_indemnification_object`, `is_indemnification_indobject`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/legal/models/legre_indemnifications_md_en_1.0.0_3.0_1668012403648.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/legal/models/legre_indemnifications_md_en_1.0.0_3.0_1668012403648.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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
    .setMaxSyntacticDistance(5)\
    .setRelationPairs(['INDEMNIFICATION_SUBJECT-INDEMNIFICATION_ACTION', 'INDEMNIFICATION_SUBJECT-INDEMNIFICATION_INDIRECT_OBJECT', 'INDEMNIFICATION_ACTION-INDEMNIFICATION'])
# =================

reDL = legal.RelationExtractionDLModel.pretrained("legre_indemnifications_md", "en", "legal/models") \
    .setPredictionThreshold(0.9) \
    .setInputCols(["re_ner_chunks", "sentence"]) \
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
model = pipeline.fit(data)
res = model.transform(data)
```

</div>

## Results

```bash
+----------------------------+-----------------------+-------------+-----------+---------------+-------------------------------+-------------+-----------+-----------+----------+
|relation                    |entity1                |entity1_begin|entity1_end|chunk1         |entity2                        |entity2_begin|entity2_end|chunk2     |confidence|
+----------------------------+-----------------------+-------------+-----------+---------------+-------------------------------+-------------+-----------+-----------+----------+
|is_indemnification_subject  |INDEMNIFICATION_ACTION |12           |26         |shall indemnify|INDEMNIFICATION_SUBJECT        |4            |10         |Company    |0.9905861 |
|is_indemnification_subject  |INDEMNIFICATION_ACTION |32           |44         |hold harmless  |INDEMNIFICATION_SUBJECT        |4            |10         |Company    |0.9996145 |
|is_indemnification_indobject|INDEMNIFICATION_SUBJECT|4            |10         |Company        |INDEMNIFICATION_INDIRECT_OBJECT|46           |48         |HOC        |0.9948344 |
|is_indemnification_object   |INDEMNIFICATION_ACTION |12           |26         |shall indemnify|INDEMNIFICATION                |58           |67         |any losses |0.9983841 |
|is_indemnification_object   |INDEMNIFICATION_ACTION |12           |26         |shall indemnify|INDEMNIFICATION                |70           |75         |claims     |0.9972869 |
|is_indemnification_object   |INDEMNIFICATION_ACTION |12           |26         |shall indemnify|INDEMNIFICATION                |78           |84         |damages    |0.99586076|
|is_indemnification_object   |INDEMNIFICATION_ACTION |12           |26         |shall indemnify|INDEMNIFICATION                |89           |99         |liabilities|0.9969894 |
|is_indemnification_object   |INDEMNIFICATION_ACTION |32           |44         |hold harmless  |INDEMNIFICATION                |58           |67         |any losses |0.9989536 |
|is_indemnification_object   |INDEMNIFICATION_ACTION |32           |44         |hold harmless  |INDEMNIFICATION                |70           |75         |claims     |0.99755704|
|is_indemnification_object   |INDEMNIFICATION_ACTION |32           |44         |hold harmless  |INDEMNIFICATION                |78           |84         |damages    |0.99725854|
|is_indemnification_object   |INDEMNIFICATION_ACTION |32           |44         |hold harmless  |INDEMNIFICATION                |89           |99         |liabilities|0.997675  |
+----------------------------+-----------------------+-------------+-----------+---------------+-------------------------------+-------------+-----------+-----------+----------+

```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|legre_indemnifications_md|
|Compatibility:|Legal NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|
|Size:|402.3 MB|

## References

Manual annotations on CUAD dataset

## Benchmarking

```bash

Relation                        Recall    Precision  F1       Support

is_indemnification_indobject     1.000     1.000     1.000        18
is_indemnification_object        0.972     1.000     0.986        36
is_indemnification_subject       0.800     0.800     0.800        10
other                            0.972     0.946     0.959        36

Avg.                0.936     0.936     0.936

Weighted Avg.       0.960     0.961     0.960

```
