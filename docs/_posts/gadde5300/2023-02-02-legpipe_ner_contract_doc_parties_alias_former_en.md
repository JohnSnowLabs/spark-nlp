---
layout: model
title: Legal NER Pipeline(Parties, Dates, Document Type)
author: John Snow Labs
name: legpipe_ner_contract_doc_parties_alias_former
date: 2023-02-02
tags: [legal, bert, licensed, agreements, en]
task: Named Entity Recognition
language: en
nav_key: models
edition: Legal NLP 1.0.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

IMPORTANT: Don't run this pretrained pipeline on the whole legal agreement. Instead:
- Split by paragraphs. You can use [notebook 1](https://github.com/JohnSnowLabs/spark-nlp-workshop/tree/master/tutorials/Certification_Trainings_JSL) in Finance or Legal as inspiration;
- Use the `legclf_introduction_clause` Text Classifier to select only these paragraphs; 

This is a Legal NER Pipeline, aimed to process the first page of the agreements when information can be found about:
- Parties of the contract/agreement;
- Aliases of those parties, or how those parties will be called further on in the document;
- Document Type;
- Effective Date of the agreement;

This pretrained pipeline can be used all along with its Relation Extraction model to retrieve the relations between these entities, called `legre_contract_doc_parties`

Other models can be found to detect other parts of the document, as Headers/Subheaders, Signers, "Will-do", etc.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/legal/models/legpipe_ner_contract_doc_parties_alias_former_en_1.0.0_3.0_1675360136179.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/legal/models/legpipe_ner_contract_doc_parties_alias_former_en_1.0.0_3.0_1675360136179.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
legal_pipeline = nlp.PretrainedPipeline("legpipe_ner_contract_doc_parties_alias_former", "en", "legal/models")

text = ['''This Consulting Agreement (the "Agreement"), made this 27t h day of March, 2017 is entered into by Immunotolerance, Inc., a Delaware corporation (the "Company"), and Alan Crane, an individual (the "Consultant").''']

result = legal_pipeline.annotate(text)
```

</div>

## Results

```bash

+------------------------+---------+
|chunk                   |ner_label|
+------------------------+---------+
|Consulting Agreement    |DOC      |
|"Agreement"             |ALIAS    |
|27t h day of March, 2017|EFFDATE  |
|Immunotolerance         |PARTY    |
|"Company"               |ALIAS    |
|Alan Crane              |PARTY    |
|"Consultant"            |ALIAS    |
+------------------------+---------+

```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|legpipe_ner_contract_doc_parties_alias_former|
|Type:|pipeline|
|Compatibility:|Legal NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|
|Size:|949.3 MB|

## Included Models

- DocumentAssembler
- SentenceDetector
- TokenizerModel
- ContextualParserModel
- ContextualParserModel
- RoBertaEmbeddings
- LegalNerModel
- NerConverterInternalModel
- ZeroShotNerModel
- NerConverterInternalModel
- ChunkMergeModel
