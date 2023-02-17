---
layout: model
title: Legal Relation Extraction (Parties, Alias, Dates, Document Type) (Lg, Unidirectional)
author: John Snow Labs
name: legre_contract_doc_parties_lg
date: 2023-02-17
tags: [legal, licensed, agreements, en, tensorflow]
task: Relation Extraction
language: en
edition: Legal NLP 1.0.0
spark_version: 3.0
supported: true
engine: tensorflow
annotator: RelationExtractionDLModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

IMPORTANT: Don't run this model on the whole legal agreement. Instead:
- Split by paragraphs. You can use [notebook 1](https://github.com/JohnSnowLabs/spark-nlp-workshop/tree/master/tutorials/Certification_Trainings_JSL) in Finance or Legal as inspiration;
- Use the `legclf_introduction_clause` Text Classifier to select only these paragraphs; 
- 
This is a Legal Relation Extraction model, which can be used after the NER Model for extracting Parties, Document Types, Effective Dates and Aliases, called legner_contract_doc_parties.

As an output, you will get the relations linking the different concepts together, if such relation exists. The list of relations is:

- dated_as: A Document has an Effective Date
- has_alias: The Alias of a Party all along the document
- has_collective_alias: An Alias hold by several parties at the same time
- signed_by: Between a Party and the document they signed

This is a `lg` model with Unidirectional Relations, meaning that the model retrieves in chunk1 the left side of the relation (source), and in chunk2 the right side (target).

## Predicted Entities

`dated_as`, `has_alias`, `has_collective_alias`, `signed_by`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/legal/models/legre_contract_doc_parties_lg_en_1.0.0_3.0_1676633934665.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/legal/models/legre_contract_doc_parties_lg_en_1.0.0_3.0_1676633934665.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
document_assembler = nlp.DocumentAssembler()\
  .setInputCol("text")\
  .setOutputCol("document")

sen = nlp.SentenceDetector()\
    .setInputCols(["document"])\
    .setOutputCol("sentence")

tokenizer = nlp.Tokenizer()\
    .setInputCols("sentence")\
    .setOutputCol("token")

embeddings = nlp.RoBertaEmbeddings.pretrained("roberta_embeddings_legal_roberta_base", "en") \
    .setInputCols("sentence", "token")\
    .setOutputCol("embeddings")\
    .setMaxSentenceLength(512)

pos_tagger = nlp.PerceptronModel()\
    .pretrained("pos_clinical", "en", "clinical/models") \
    .setInputCols(["sentence", "token"])\
    .setOutputCol("pos_tags")
    
dependency_parser = nlp.DependencyParserModel()\
    .pretrained("dependency_conllu", "en")\
    .setInputCols(["sentence", "pos_tags", "token"])\
    .setOutputCol("dependencies")

ner_model = legal.NerModel.pretrained('legner_contract_doc_parties_lg', 'en', 'legal/models')\
    .setInputCols(["sentence", "token", "embeddings"])\
    .setOutputCol("ner")

ner_converter = nlp.NerConverter()\
    .setInputCols(["sentence","token","ner"])\
    .setOutputCol("ner_chunk")

re_ner_chunk_filter = legal.RENerChunksFilter() \
    .setInputCols(["ner_chunks", "dependencies"])\
    .setOutputCol("re_ner_chunks")\
    .setMaxSyntacticDistance(7)\
    .setRelationPairs(["DOC-EFFDATE", "DOC-PARTY", "PARTY-FORMER_NAME", "PARTY-ALIAS"])

re_model = legal.RelationExtractionDLModel().pretrained('legre_contract_doc_parties_lg', 'en', 'legal/models')\
    .setPredictionThreshold(0.5)\
    .setInputCols(["re_ner_chunks", "sentence"])\
    .setOutputCol("relations")

nlpPipeline = nlp.Pipeline(stages=[
    document_assembler,
    sen,
    tokenizer,
    embeddings,
    pos_tagger,
    dependency_parser,
    ner_model,
    ner_converter,
    re_ner_chunk_filter,
    re_model
    ])

empty_df = spark.createDataFrame([[""]]).toDF("text")

model = nlpPipeline.fit(empty_df)

text="""THIS Lease Agreement ,  is made and  entered  into this  _____day of May,  2006 by and between Apple, Inc.,  (hereinafter called "Landlord"),  and IMI Global,  Inc., with a mailing address of ___,  (hereinafter referred as "Tenant")."""

data = spark.createDataFrame([[text]]).toDF("text")

result = model.transform(data)
```

</div>

## Results

```bash

+---------+-----------------+--------------------+-----------------+----------------+----------+------------------+
|relations|relations_entity1|    relations_chunk1|relations_entity2|relations_chunk2|confidence|syntactic_distance|
+---------+-----------------+--------------------+-----------------+----------------+----------+------------------+
| dated_as|              DOC|THIS Lease Agreement|          EFFDATE|   of May,  2006| 0.9999546|                 6|
|signed_by|              DOC|THIS Lease Agreement|            PARTY|      Apple, Inc|  0.988555|                 5|
|signed_by|              DOC|THIS Lease Agreement|            PARTY|IMI Global,  Inc| 0.9568861|                 7|
|has_alias|            PARTY|          Apple, Inc|            ALIAS|        Landlord|0.99999475|                 4|
|has_alias|            PARTY|    IMI Global,  Inc|            ALIAS|          Tenant| 0.9999893|                 4|
+---------+-----------------+--------------------+-----------------+----------------+----------+------------------+

```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|legre_contract_doc_parties_lg|
|Compatibility:|Legal NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|
|Size:|406.0 MB|

## Benchmarking

```bash
              
Label                         Recall  Precision     F1   Support
dated_as                      1.000     1.000     1.000     19
has_alias                     1.000     1.000     1.000     29
has_collective_alias          1.000     1.000     1.000     25
signed_by                     1.000     1.000     1.000     47
Avg.                          1.000     1.000     1.000     -
Weighted-Avg.                 1.000     1.000     1.000     -
```
