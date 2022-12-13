---
layout: model
title: Legal Relation Extraction (Grants, md, Unidirectional)
author: John Snow Labs
name: legre_grants_md
date: 2022-11-09
tags: [en, legal, licensed, grants, re]
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

This model requires `legner_bert_grants` as an NER in the pipeline. It's a `md` model with Unidirectional Relations, meaning that the model retrieves in chunk1 the left side of the relation (source), and in chunk2 the right side (target).

## Predicted Entities

`allows`, `is_allowed_to`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/legal/models/legre_grants_md_en_1.0.0_3.0_1668017439874.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/legal/models/legre_grants_md_en_1.0.0_3.0_1668017439874.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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
    
ner_model = legal.BertForTokenClassification.pretrained("legner_bert_grants", "en", "legal/models")\
  .setInputCols("token", "sentence")\
  .setOutputCol("ner")\
  .setCaseSensitive(True)
        
ner_converter = nlp.NerConverter() \
        .setInputCols(["sentence","token","ner"]) \
        .setOutputCol("ner_chunk")
        
re_filter = legal.RENerChunksFilter()\
    .setInputCols(["ner_chunk", "dependencies"])\
    .setOutputCol("re_ner_chunks")\
    .setMaxSyntacticDistance(10)\
    .setRelationPairs(['PERMISSION_SUBJECT-PERMISSION_INDIRECT_OBJECT','PERMISSION_INDIRECT_OBJECT-PERMISSION'])
    
reDL = legal.RelationExtractionDLModel.pretrained("legre_grants_md", "en", "legal/models") \
    .setPredictionThreshold(0.9) \
    .setInputCols(["re_ner_chunks", "sentence"]) \
    .setOutputCol("relations")

pipeline = Pipeline(stages=[documentAssembler,sentencizer, tokenizer,pos_tagger,dependency_parser, ner_model, ner_converter,re_filter, reDL])

text = """Appointment  Subject to payment of the Annual Minimum Commitment ("AMC"  - defined herein), Diversinet hereby grants to Reseller an exclusive, non- transferable and non-assignable right to market, sell, and sub-license those Diversinet products listed in Schedule 2 (the "Products") within the  territory listed in Schedule 3 (the "Territory") to Canadian headquartered companies, and governmental and broader public sector entities located  in Canada. """

data = spark.createDataFrame([[text]]).toDF("text")
model = pipeline.fit(data)
res = model.transform(data)
```

</div>

## Results

```bash
+-------------+--------------------------+-------------+-----------+----------+--------------------------+-------------+-----------+----------------------------------------------------------------------------+----------+
|relation     |entity1                   |entity1_begin|entity1_end|chunk1    |entity2                   |entity2_begin|entity2_end|chunk2                                                                      |confidence|
+-------------+--------------------------+-------------+-----------+----------+--------------------------+-------------+-----------+----------------------------------------------------------------------------+----------+
|allows       |PERMISSION_SUBJECT        |92           |101        |Diversinet|PERMISSION_INDIRECT_OBJECT|120          |127        |Reseller                                                                    |0.99999297|
|is_allowed_to|PERMISSION_INDIRECT_OBJECT|120          |127        |Reseller  |PERMISSION                |132          |145        |exclusive, non                                                              |0.9999945 |
|is_allowed_to|PERMISSION_INDIRECT_OBJECT|120          |127        |Reseller  |PERMISSION                |148          |223        |transferable and non-assignable right to market, sell, and sub-license those|0.99987125|
+-------------+--------------------------+-------------+-----------+----------+--------------------------+-------------+-----------+----------------------------------------------------------------------------+----------+

```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|legre_grants_md|
|Compatibility:|Legal NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|
|Size:|402.2 MB|

## References

Manual annotations on CUAD dataset

## Benchmarking

```bash

Relation           Recall Precision        F1   Support

allows              1.000     1.000     1.000        32
is_allowed_to       1.000     1.000     1.000        36
other               1.000     1.000     1.000        32

Avg.                1.000     1.000     1.000

Weighted Avg.       1.000     1.000     1.000

```
