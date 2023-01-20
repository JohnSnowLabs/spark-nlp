---
layout: model
title: Finance Pipeline (Headers / Subheaders)
author: John Snow Labs
name: finpipe_header_subheader
date: 2023-01-20
tags: [en, finance, ner, licensed, contextual_parser]
task: Named Entity Recognition
language: en
edition: Finance NLP 1.0.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This is a finance pretrained pipeline that will help you split long financial documents into smaller sections. To do that, it detects Headers and Subheaders of different sections. You can then use the beginning and end information in the metadata to retrieve the text between those headers.

PART I, PART II, etc are HEADERS
Item 1, Item 2, etc are also HEADERS
Item 1A, 2B, etc are SUBHEADERS
1., 2., 2.1, etc. are SUBHEADERS

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/finance/models/finpipe_header_subheader_en_1.0.0_3.0_1674243435691.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/finance/models/finpipe_header_subheader_en_1.0.0_3.0_1674243435691.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
finance_pipeline = nlp.PretrainedPipeline("finpipe_header_subheader", "en", "finance/models")

text = ["""
Item 2. Definitions. 
For purposes of this Agreement, the following terms have the meanings ascribed thereto in this Section 1. 2. Appointment as Reseller.

Item 2A. Appointment. 
The Company hereby [***]. Allscripts may also disclose Company's pricing information relating to its Merchant Processing Services and facilitate procurement of Merchant Processing Services on behalf of Sublicensed Customers, including, without limitation by references to such pricing information and Merchant Processing Services in Customer Agreements. 6

Item 2B. Customer Agreements."""]

result = finance_pipeline.annotate(text)
```

</div>

## Results

```bash
|                        chunks | begin | end |  entities |
|------------------------------:|------:|----:|----------:|
|          Item 2. Definitions. |     1 |  21 |    HEADER |
|         Item 2A. Appointment. |   158 | 179 | SUBHEADER |
| Item 2B. Customer Agreements. |   538 | 566 | SUBHEADER |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|finpipe_header_subheader|
|Type:|pipeline|
|Compatibility:|Finance NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|
|Size:|23.6 KB|

## Included Models

- DocumentAssembler
- TokenizerModel
- ContextualParserModel
- ContextualParserModel
- ChunkMergeModel
