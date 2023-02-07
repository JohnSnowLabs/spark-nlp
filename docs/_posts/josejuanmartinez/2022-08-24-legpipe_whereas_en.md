---
layout: model
title: Whereas Pipeline
author: John Snow Labs
name: legpipe_whereas
date: 2022-08-24
tags: [en, legal, whereas, licensed]
task: [Named Entity Recognition, Part of Speech Tagging, Dependency Parser, Relation Extraction]
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
- Use the `legclf_cuad_whereas_clause` Text Classifier to select only these paragraphs; 

This is a Pretrained Pipeline to show extraction of whereas clauses (Subject, Action and Object), and also the relationships between them, using two approaches:
- A Semantic Relation Extraction Model;
- A Dependency Parser Tree;

The difficulty of these entities is that they are totally free-text, with OBJECT being sometimes very long with very diverse vocabulary. Although the NER and the REDL can help you identify them, the Dependency Parser has been added so you can navigate the tree looking for specific direct objects or other phrases.

## Predicted Entities

`WHEREAS_SUBJECT`, `WHEREAS_OBJECT`, `WHEREAS_ACTION`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/finance/LEGALNER_WHEREAS){:.button.button-orange}
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/legal/models/legpipe_whereas_en_1.0.0_3.2_1661340138139.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/legal/models/legpipe_whereas_en_1.0.0_3.2_1661340138139.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
from johnsnowlabs import *

deid_pipeline = PretrainedPipeline("legpipe_whereas", "en", "legal/models")

deid_pipeline.annotate('WHEREAS VerticalNet owns and operates a series of online communities.')

# Return NER chunks
pipeline_result['ner_chunk']

# Return RE
pipeline_result['relations']

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
['VerticalNet', 'operates', 'a series of online communities']

# Relations
['has_subject', 'has_subject', 'has_object']

# DEP
# Use Spark NLP Display to see the dependency tree
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|legpipe_whereas|
|Type:|pipeline|
|Compatibility:|Legal NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|
|Size:|918.6 MB|

## References

In-house annotations on CUAD dataset

## Included Models

- nlp.DocumentAssembler
- nlp.Tokenizer
- nlp.PerceptronModel
- nlp.DependencyParserModel
- nlp.TypedDependencyParserModel
- nlp.RoBertaEmbeddings
- legal.NerModel
- nlp.NerConverter
- legal.RelationExtractionDLModel