---
layout: model
title: Financial Relation Extraction on German Financial Statements
author: John Snow Labs
name: finre_has_value
date: 2023-03-25
tags: [re, licensed, finance, de, tensorflow]
task: Relation Extraction
language: de
edition: Finance NLP 1.0.0
spark_version: 3.0
supported: true
engine: tensorflow
annotator: RelationExtractionDLModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model is a relation extraction model to extract financial entities and their values from text with `finner_financial_entity_value` model.

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/finance/models/finre_has_value_de_1.0.0_3.0_1679702750286.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/finance/models/finre_has_value_de_1.0.0_3.0_1679702750286.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = nlp.DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

sen = nlp.SentenceDetectorDLModel.pretrained("sentence_detector_dl","xx")\
        .setInputCols(["document"])\
        .setOutputCol("sentence")

tokenizer = nlp.Tokenizer()\
    .setInputCols("sentence")\
    .setOutputCol("token")

embeddings = nlp.BertEmbeddings.pretrained("bert_sentence_embeddings_financial", "de")\
    .setInputCols("document", "token")\
    .setOutputCol("embeddings")\
    .setMaxSentenceLength(512)\
    .setCaseSensitive(True)

pos_tagger = nlp.PerceptronModel()\
    .pretrained("pos_clinical", "en", "clinical/models") \
    .setInputCols(["sentence", "token"])\
    .setOutputCol("pos_tags")
    
dependency_parser = nlp.DependencyParserModel()\
    .pretrained("dependency_conllu", "en")\
    .setInputCols(["sentence", "pos_tags", "token"])\
    .setOutputCol("dependencies")
    
ner_model = finance.NerModel().pretrained('finner_financial_entity_value', 'de', 'finance/models)\
        .setInputCols(["sentence", "token", "embeddings"])\
        .setOutputCol("ner1")

ner_converter = nlp.NerConverter()\
    .setInputCols(["sentence","token","ner1"])\
    .setOutputCol("ner_chunks")

re_ner_chunk_filter = finance.RENerChunksFilter() \
    .setInputCols(["ner_chunks", "dependencies"])\
    .setOutputCol("re_ner_chunks")\
    .setRelationPairs(["FINANCIAL_ENTITY-FINANCIAL_VALUE", "FINANCIAL_VALUE-FINANCIAL_ENTITY"])

reDL = finance.RelationExtractionDLModel().pretrained('finre_has_value', 'de', 'finance/models)\
    .setPredictionThreshold(0.5)\
    .setInputCols(["re_ner_chunks", "sentence"])\
    .setOutputCol("relations")
   

nlpPipeline = nlp.Pipeline(stages=[
    documentAssembler,
    sen,
    tokenizer,
    embeddings,
    pos_tagger,
    dependency_parser,
    ner_model,
    ner_converter,
    re_ner_chunk_filter,
    reDL
])


empty_df = spark.createDataFrame([[""]]).toDF("text")

model = nlpPipeline.fit(empty_df)

text= """Die Darlehensverbindlichkeit gegenüber der Vaillant GmbH in Höhe von TEUR 3.433 hat eine Laufzeit bis zum 31.12.2019 , die restlichen Verbindlichkeiten haben eine Restlaufzeit bis zu einem Jahr ."""
sdf = spark.createDataFrame([[text]]).toDF("text")

res = model.transform(sdf)
res.show(20,truncate=False)

result_df = res.select(F.explode(F.arrays_zip(res.relations.result, 
                                                 res.relations.metadata)).alias("cols")) \
                  .select(
                          F.expr("cols['0']").alias("relations"),\
                          F.expr("cols['1']['entity1']").alias("relations_entity1"),\
                          F.expr("cols['1']['chunk1']" ).alias("relations_chunk1" ),\
                          F.expr("cols['1']['entity2']").alias("relations_entity2"),\
                          F.expr("cols['1']['chunk2']" ).alias("relations_chunk2" ),\
                          F.expr("cols['1']['confidence']" ).alias("confidence" ),\
                          F.expr("cols['1']['syntactic_distance']" ).alias("syntactic_distance" ),\
                          ).filter("relations!='other'")

result_df.show()
```

</div>

## Results

```bash

relations relations_entity1     relations_chunk1 relations_entity2  relations_chunk2 confidence syntactic_distance 
has_value  FINANCIAL_ENTITY Darlehensverbindl...   FINANCIAL_VALUE             3.433        1.0                  8 
has_value   FINANCIAL_VALUE                3.433  FINANCIAL_ENTITY Verbindlichkeiten        1.0          undefined 
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|finre_has_value|
|Compatibility:|Finance NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Language:|de|
|Size:|661.2 MB|

## Benchmarking

```bash
Relation           Recall Precision        F1   Support
has_value           1.000     1.000     1.000       100
Avg.                1.000     1.000     1.000         -
Weighted Avg.       1.000     1.000     1.000         -
```