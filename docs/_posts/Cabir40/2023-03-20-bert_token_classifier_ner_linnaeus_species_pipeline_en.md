---
layout: model
title: Pipeline to Detect Organism in Medical Texts
author: John Snow Labs
name: bert_token_classifier_ner_linnaeus_species_pipeline
date: 2023-03-20
tags: [en, ner, clinical, licensed, bertfortokenclassification]
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

This pretrained pipeline is built on the top of [bert_token_classifier_ner_linnaeus_species](https://nlp.johnsnowlabs.com/2022/07/25/bert_token_classifier_ner_linnaeus_species_en_3_0.html) model.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/bert_token_classifier_ner_linnaeus_species_pipeline_en_4.3.0_3.2_1679303734578.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/bert_token_classifier_ner_linnaeus_species_pipeline_en_4.3.0_3.2_1679303734578.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline

pipeline = PretrainedPipeline("bert_token_classifier_ner_linnaeus_species_pipeline", "en", "clinical/models")

text = '''First identified in chicken, vigilin homologues have now been found in human (6), Xenopus laevis (7), Drosophila melanogaster (8) and Schizosaccharomyces pombe.'''

result = pipeline.fullAnnotate(text)
```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val pipeline = new PretrainedPipeline("bert_token_classifier_ner_linnaeus_species_pipeline", "en", "clinical/models")

val text = "First identified in chicken, vigilin homologues have now been found in human (6), Xenopus laevis (7), Drosophila melanogaster (8) and Schizosaccharomyces pombe."

val result = pipeline.fullAnnotate(text)
```
</div>

## Results

```bash
|    | ner_chunk                 |   begin |   end | ner_label   |   confidence |
|---:|:--------------------------|--------:|------:|:------------|-------------:|
|  0 | chicken                   |      20 |    26 | SPECIES     |     0.998697 |
|  1 | human                     |      71 |    75 | SPECIES     |     0.999767 |
|  2 | Xenopus laevis            |      82 |    95 | SPECIES     |     0.999918 |
|  3 | Drosophila melanogaster   |     102 |   124 | SPECIES     |     0.999925 |
|  4 | Schizosaccharomyces pombe |     134 |   158 | SPECIES     |     0.999881 |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_token_classifier_ner_linnaeus_species_pipeline|
|Type:|pipeline|
|Compatibility:|Healthcare NLP 4.3.0+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|
|Size:|404.7 MB|

## Included Models

- DocumentAssembler
- SentenceDetectorDLModel
- TokenizerModel
- MedicalBertForTokenClassifier
- NerConverterInternalModel