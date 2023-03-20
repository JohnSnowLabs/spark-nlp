---
layout: model
title: Pipeline to Detect Genes and Human Phenotypes (biobert)
author: John Snow Labs
name: ner_human_phenotype_gene_biobert_pipeline
date: 2023-03-20
tags: [ner, clinical, licensed, en]
task: Named Entity Recognition
language: en
edition: Healthcare NLP 4.3.0
spark_version: 3.2
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This pretrained pipeline is built on the top of [ner_human_phenotype_gene_biobert](https://nlp.johnsnowlabs.com/2021/04/01/ner_human_phenotype_gene_biobert_en.html) model.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_human_phenotype_gene_biobert_pipeline_en_4.3.0_3.2_1679315678860.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/ner_human_phenotype_gene_biobert_pipeline_en_4.3.0_3.2_1679315678860.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline

pipeline = PretrainedPipeline("ner_human_phenotype_gene_biobert_pipeline", "en", "clinical/models")

text = '''Here we presented a case (BS type) of a 17 years old female presented with polyhydramnios, polyuria, nephrocalcinosis and hypokalemia, which was alleviated after treatment with celecoxib and vitamin D(3).'''

result = pipeline.fullAnnotate(text)
```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val pipeline = new PretrainedPipeline("ner_human_phenotype_gene_biobert_pipeline", "en", "clinical/models")

val text = "Here we presented a case (BS type) of a 17 years old female presented with polyhydramnios, polyuria, nephrocalcinosis and hypokalemia, which was alleviated after treatment with celecoxib and vitamin D(3)."

val result = pipeline.fullAnnotate(text)
```
</div>

## Results

```bash
|    | ner_chunk        |   begin |   end | ner_label   |   confidence |
|---:|:-----------------|--------:|------:|:------------|-------------:|
|  0 | type             |      29 |    32 | GENE        |       0.9977 |
|  1 | polyhydramnios   |      75 |    88 | HP          |       0.9949 |
|  2 | polyuria         |      91 |    98 | HP          |       0.9955 |
|  3 | nephrocalcinosis |     101 |   116 | HP          |       0.995  |
|  4 | hypokalemia      |     122 |   132 | HP          |       0.9986 |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_human_phenotype_gene_biobert_pipeline|
|Type:|pipeline|
|Compatibility:|Healthcare NLP 4.3.0+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|
|Size:|422.2 MB|

## Included Models

- DocumentAssembler
- SentenceDetectorDLModel
- TokenizerModel
- BertEmbeddings
- MedicalNerModel
- NerConverterInternalModel