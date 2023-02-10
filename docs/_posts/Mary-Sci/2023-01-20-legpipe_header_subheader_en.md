---
layout: model
title: Legal Pipeline (Headers / Subheaders)
author: John Snow Labs
name: legpipe_header_subheader
date: 2023-01-20
tags: [en, licensed, legal, ner, contextual_parser]
task: Named Entity Recognition
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

This is a Legal pretrained pipeline, aimed to carry out Section Splitting by using the Headers and Subheaders entities, detected in the document.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/legal/models/legpipe_header_subheader_en_1.0.0_3.0_1674244247295.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/legal/models/legpipe_header_subheader_en_1.0.0_3.0_1674244247295.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
legal_pipeline = nlp.PretrainedPipeline("legpipe_header_subheader", "en", "legal/models")

text = ["""2. DEFINITION. 
For purposes of this Agreement, the following terms have the meanings ascribed thereto in this Section 1 and 2 Appointment as Reseller.
2.1 Appointment. 
The Company hereby [***]. Allscripts may also disclose Company's pricing information relating to its Merchant Processing Services and facilitate procurement of Merchant Processing Services on behalf of Sublicensed Customers, including, without limitation by references to such pricing information and Merchant Processing Services in Customer Agreements. 6
2.2 Customer Agreements."""]

result = legal_pipeline.annotate(text)
```

</div>

## Results

```bash
|                  chunks | begin | end |  entities |
|------------------------:|------:|----:|----------:|
|           2. DEFINITION |     0 |  12 |    HEADER |
|         2.1 Appointment |   154 | 168 | SUBHEADER |
| 2.2 Customer Agreements |   530 | 552 | SUBHEADER |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|legpipe_header_subheader|
|Type:|pipeline|
|Compatibility:|Legal NLP 1.0.0+|
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
