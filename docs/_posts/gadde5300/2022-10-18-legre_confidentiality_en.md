---
layout: model
title: Legal Relation Extraction (Confidentiality, sm, Bidirectional)
author: John Snow Labs
name: legre_confidentiality
date: 2022-10-18
tags: [legal, en, re, licensed, confidentiality]
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

This is a Legal Relation Extraction Model to identify the Subject (who), Action (web), Object(the indemnification) and Indirect Object (to whom) from confidentiality clauses.

This model is a `sm` model without meaningful directions in the relations (the model was not trained to understand if the direction of the relation is from left to right or right to left).

There are bigger models in Models Hub trained also with directed relationships.

## Predicted Entities

`is_confidentiality_indobject`, `is_confidentiality_object`, `is_confidentiality_subject`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/legal/models/legre_confidentiality_en_1.0.0_3.0_1666098845071.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/legal/models/legre_confidentiality_en_1.0.0_3.0_1666098845071.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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
    
reDL = legal.RelationExtractionDLModel.pretrained("legre_confidentiality", "en", "legal/models") \
    .setPredictionThreshold(0.5) \
    .setInputCols(["re_ner_chunks", "sentence"]) \
    .setOutputCol("relations")
    
pipeline = Pipeline(stages=[documentAssembler,sentencizer, tokenizer,pos_tagger,dependency_parser, embeddings, ner_model, ner_converter,re_filter, reDL])
text = "Each party will promptly return to the other upon request any Confidential Information of the other party then in its possession or under its control."
data = spark.createDataFrame([[text]]).toDF("text")
model = pipeline.fit(data)
res = model.transform(data)
```

</div>

## Results

```bash

+----------------------------+-----------------------+-------------+-----------+--------------------+-------------------------------+-------------+-----------+------------------------+----------+
|relation                    |entity1                |entity1_begin|entity1_end|chunk1              |entity2                        |entity2_begin|entity2_end|chunk2                  |confidence|
+----------------------------+-----------------------+-------------+-----------+--------------------+-------------------------------+-------------+-----------+------------------------+----------+
|is_confidentiality_subject  |CONFIDENTIALITY_SUBJECT|0            |9          |Each party          |CONFIDENTIALITY_ACTION         |11           |30         |will promptly return    |0.9745122 |
|is_confidentiality_indobject|CONFIDENTIALITY_SUBJECT|0            |9          |Each party          |CONFIDENTIALITY_INDIRECT_OBJECT|39           |43         |other                   |0.89561754|
|is_confidentiality_object   |CONFIDENTIALITY_ACTION |11           |30         |will promptly return|CONFIDENTIALITY                |62           |85         |Confidential Information|0.9981041 |
+----------------------------+-----------------------+-------------+-----------+--------------------+-------------------------------+-------------+-----------+------------------------+----------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|legre_confidentiality|
|Compatibility:|Legal NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|
|Size:|409.9 MB|

## References

In-house annotated examples from CUAD legal dataset

## Benchmarking

```bash
Relation                        Recall    Precision     F1   Support
is_confidentiality_indobject    1.000     1.000      1.000        28
is_confidentiality_object       1.000     0.981      0.990        51
is_confidentiality_subject      0.970     1.000      0.985        33
Avg.                            0.990     0.994     0.992
Weighted Avg.                   0.991     0.991     0.991
```
