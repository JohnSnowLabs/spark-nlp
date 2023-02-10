---
layout: model
title: Obligations Pipeline
author: John Snow Labs
name: legpipe_obligations
date: 2022-08-24
tags: [en, legal, obligations, licensed]
task: [Named Entity Recognition, Part of Speech Tagging, Dependency Parser]
language: en
edition: Legal NLP 1.0.0
spark_version: 3.0
supported: true
annotator: Pipeline
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

IMPORTANT: Don't run this model on the whole legal agreement. Instead:
- Split by paragraphs. You can use [notebook 1](https://github.com/JohnSnowLabs/spark-nlp-workshop/tree/master/tutorials/Certification_Trainings_JSL) in Finance or Legal as inspiration;
- Use the `legclf_cuad_obligations_clause` Text Classifier to select only these paragraphs; 

This is a Pretrained Pipeline to process agreements, more specifically the sentences where all the obligations of the parties are expressed (what they agreed upon in the contract).

This pipeline returns:
- NER entities for the subject, the action/verb, the object and the indirect object of the clause;
- Syntactic dependencies of the chunks, so that you can disambiguate in case different clauses/agreements are present in the same sentence.

This model does not include a Sentence Detector, it executes everything at document-level. If you want to split by sentences, do it before and call this pipeline with the text of the sentences.

## Predicted Entities

`OBLIGATION_SUBJECT`, `OBLIGATION_ACTION`, `OBLIGATION`, `OBLIGATION_INDIRECT_OBJECT`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/legal/models/legpipe_obligations_en_1.0.0_3.2_1661342149969.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/legal/models/legpipe_obligations_en_1.0.0_3.2_1661342149969.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
from johnsnowlabs import *

deid_pipeline = PretrainedPipeline("legpipe_obligations", "en", "legal/models")

deid_pipeline.annotate('The Supplier agrees to provide the Buyer with all the necessary documents to fulfill the agreement')

# Return NER chunkcs
pipeline_result['ner_chunk']

# Visualize the Dependencies

dependency_vis = viz.DependencyParserVisualizer()

dependency_vis.display(pipeline_result[0], #should be the results of a single example, not the complete dataframe.
                       pos_col = 'pos', #specify the pos column
                       dependency_col = 'dependencies', #specify the dependency column
                       dependency_type_col = 'dependency_type' #specify the dependency type column
                       )
```

</div>

## Results

```bash
# NER

['Supplier',
 'agrees to provide',
 'Buyer',
 'with all the necessary documents to fulfill the agreement']

# DEP
# Use Spark NLP Display to see the dependency tree
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|legpipe_obligations|
|Type:|pipeline|
|Compatibility:|Legal NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|
|Size:|435.9 MB|

## References

In-house annotations on CUAD dataset

## Included Models

- nlp.DocumentAssembler
- nlp.Tokenizer
- nlp.PerceptronModel
- nlp.DependencyParserModel
- nlp.TypedDependencyParserModel
- legal.BertForTokenClassification
- nlp.NerConverter