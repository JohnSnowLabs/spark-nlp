---
layout: model
title: Pipeline to Detect Genes/Proteins (BC2GM) in Medical Texts
author: John Snow Labs
name: ner_biomedical_bc2gm_pipeline
date: 2023-03-14
tags: [bc2gm, ner, biomedical, gene_protein, gene, protein, en, licensed, clinical]
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

This pretrained pipeline is built on the top of [ner_biomedical_bc2gm](https://nlp.johnsnowlabs.com/2022/05/11/ner_biomedical_bc2gm_en_2_4.html) model.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_biomedical_bc2gm_pipeline_en_4.3.0_3.2_1678787724252.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/ner_biomedical_bc2gm_pipeline_en_4.3.0_3.2_1678787724252.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline

pipeline = PretrainedPipeline("ner_biomedical_bc2gm_pipeline", "en", "clinical/models")

text = '''Immunohistochemical staining was positive for S-100 in all 9 cases stained, positive for HMB-45 in 9 (90%) of 10, and negative for cytokeratin in all 9 cases in which myxoid melanoma remained in the block after previous sections.'''

result = pipeline.fullAnnotate(text)
```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val pipeline = new PretrainedPipeline("ner_biomedical_bc2gm_pipeline", "en", "clinical/models")

val text = "Immunohistochemical staining was positive for S-100 in all 9 cases stained, positive for HMB-45 in 9 (90%) of 10, and negative for cytokeratin in all 9 cases in which myxoid melanoma remained in the block after previous sections."

val result = pipeline.fullAnnotate(text)
```
</div>

## Results

```bash
|    | ner_chunks   |   begin |   end | ner_label    |   confidence |
|---:|:-------------|--------:|------:|:-------------|-------------:|
|  0 | S-100        |      46 |    50 | GENE_PROTEIN |       0.9911 |
|  1 | HMB-45       |      89 |    94 | GENE_PROTEIN |       0.9944 |
|  2 | cytokeratin  |     131 |   141 | GENE_PROTEIN |       0.9951 |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_biomedical_bc2gm_pipeline|
|Type:|pipeline|
|Compatibility:|Healthcare NLP 4.3.0+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|
|Size:|1.7 GB|

## Included Models

- DocumentAssembler
- SentenceDetectorDLModel
- TokenizerModel
- WordEmbeddingsModel
- MedicalNerModel
- NerConverterInternalModel