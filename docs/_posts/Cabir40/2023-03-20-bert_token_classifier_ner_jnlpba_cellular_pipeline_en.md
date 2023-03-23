---
layout: model
title: Pipeline to Detect Cellular/Molecular Biology Entities
author: John Snow Labs
name: bert_token_classifier_ner_jnlpba_cellular_pipeline
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

This pretrained pipeline is built on the top of [bert_token_classifier_ner_jnlpba_cellular](https://nlp.johnsnowlabs.com/2022/07/25/bert_token_classifier_ner_jnlpba_cellular_en_3_0.html) model.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/bert_token_classifier_ner_jnlpba_cellular_pipeline_en_4.3.0_3.2_1679303520732.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/bert_token_classifier_ner_jnlpba_cellular_pipeline_en_4.3.0_3.2_1679303520732.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline

pipeline = PretrainedPipeline("bert_token_classifier_ner_jnlpba_cellular_pipeline", "en", "clinical/models")

text = '''The results suggest that activation of protein kinase C, but not new protein synthesis, is required for IL-2 induction of IFN-gamma and GM-CSF cytoplasmic mRNA. It also was observed that suppression of cytokine gene expression by these agents was independent of the inhibition of proliferation. These data indicate that IL-2 and IL-12 may have distinct signaling pathways leading to the induction of IFN-gammaand GM-CSFgene expression, andthatthe NK3.3 cell line may serve as a novel model for dissecting the biochemical and molecular events involved in these pathways. A functional T-cell receptor signaling pathway is required for p95vav activity. Stimulation of the T-cell antigen receptor ( TCR ) induces activation of multiple tyrosine kinases, resulting in phosphorylation of numerous intracellular substrates. One substrate is p95vav, which is expressed exclusively in hematopoietic and trophoblast cells..'''

result = pipeline.fullAnnotate(text)
```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val pipeline = new PretrainedPipeline("bert_token_classifier_ner_jnlpba_cellular_pipeline", "en", "clinical/models")

val text = "The results suggest that activation of protein kinase C, but not new protein synthesis, is required for IL-2 induction of IFN-gamma and GM-CSF cytoplasmic mRNA. It also was observed that suppression of cytokine gene expression by these agents was independent of the inhibition of proliferation. These data indicate that IL-2 and IL-12 may have distinct signaling pathways leading to the induction of IFN-gammaand GM-CSFgene expression, andthatthe NK3.3 cell line may serve as a novel model for dissecting the biochemical and molecular events involved in these pathways. A functional T-cell receptor signaling pathway is required for p95vav activity. Stimulation of the T-cell antigen receptor ( TCR ) induces activation of multiple tyrosine kinases, resulting in phosphorylation of numerous intracellular substrates. One substrate is p95vav, which is expressed exclusively in hematopoietic and trophoblast cells.."

val result = pipeline.fullAnnotate(text)
```
</div>

## Results

```bash
|    | ner_chunk                             |   begin |   end | ner_label   |   confidence |
|---:|:--------------------------------------|--------:|------:|:------------|-------------:|
|  0 | protein kinase C                      |      39 |    54 | protein     |     0.993263 |
|  1 | IL-2                                  |     104 |   107 | protein     |     0.969095 |
|  2 | IFN-gamma and GM-CSF cytoplasmic mRNA |     122 |   158 | RNA         |     0.998495 |
|  3 | cytokine gene                         |     202 |   214 | DNA         |     0.953537 |
|  4 | IL-2                                  |     320 |   323 | protein     |     0.999317 |
|  5 | IL-12                                 |     329 |   333 | protein     |     0.999216 |
|  6 | IFN-gammaand GM-CSFgene               |     400 |   422 | protein     |     0.995236 |
|  7 | NK3.3 cell line                       |     447 |   461 | cell_line   |     0.998958 |
|  8 | T-cell receptor                       |     583 |   597 | protein     |     0.987655 |
|  9 | p95vav                                |     633 |   638 | protein     |     0.999857 |
| 10 | T-cell antigen receptor               |     669 |   691 | protein     |     0.99891  |
| 11 | TCR                                   |     695 |   697 | protein     |     0.998049 |
| 12 | tyrosine kinases                      |     732 |   747 | protein     |     0.999636 |
| 13 | p95vav                                |     834 |   839 | protein     |     0.999842 |
| 14 | hematopoietic and trophoblast cells   |     876 |   910 | cell_type   |     0.999709 |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_token_classifier_ner_jnlpba_cellular_pipeline|
|Type:|pipeline|
|Compatibility:|Healthcare NLP 4.3.0+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|
|Size:|404.8 MB|

## Included Models

- DocumentAssembler
- SentenceDetectorDLModel
- TokenizerModel
- MedicalBertForTokenClassifier
- NerConverterInternalModel