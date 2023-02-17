---
layout: model
title: Legal Relation Extraction Pretrained Pipeline(Parties, Alias, Dates, Document Type) (Lg, Unidirectional)
author: John Snow Labs
name: legpipe_re_contract_doc_parties_alias
date: 2023-02-17
tags: [legal, licensed, agreements, en, pipeline]
task: Relation Extraction
language: en
edition: Legal NLP 1.0.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This is a Legal Relation Extraction Pretrained Pipeline to get the relations linking the different concepts together, if such relation exists. The list of relations is:

- dated_as: A Document has an Effective Date
- has_alias: The Alias of a Party all along the document
- has_collective_alias: An Alias hold by several parties at the same time
- signed_by: Between a Party and the document they signed

## Predicted Entities

`dated_as`, `has_alias`, `has_collective_alias`, `signed_by`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/legal/models/legpipe_re_contract_doc_parties_alias_en_1.0.0_3.0_1676647465198.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/legal/models/legpipe_re_contract_doc_parties_alias_en_1.0.0_3.0_1676647465198.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

legal_pipeline = nlp.PretrainedPipeline("legpipe_re_contract_doc_parties_alias", "en", "finance/models")
text = '''THIS Lease Agreement , is made and entered into this _____day of May, 2006 by and between Apple, Inc., (hereinafter called "Landlord"), and IMI Global, Inc., with a mailing address of ___, (hereinafter referred as "Tenant").'''
result = legal_pipeline.annotate(text)

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
|Model Name:|legpipe_re_contract_doc_parties_alias|
|Type:|pipeline|
|Compatibility:|Legal NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|
|Size:|910.2 MB|

## Included Models

- DocumentAssembler
- SentenceDetector
- TokenizerModel
- RoBertaEmbeddings
- PerceptronModel
- DependencyParserModel
- LegalNerModel
- NerConverter
- RENerChunksFilter
- RelationExtractionDLModel
