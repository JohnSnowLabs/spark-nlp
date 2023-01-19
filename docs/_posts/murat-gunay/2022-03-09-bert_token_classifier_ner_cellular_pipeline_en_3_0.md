---
layout: model
title: Pipeline to Detect Cellular/Molecular Biology Entities
author: John Snow Labs
name: bert_token_classifier_ner_cellular_pipeline
date: 2022-03-09
tags: [cellular, ner, bert_for_token_classifier, en, licensed, clinical]
task: Named Entity Recognition
language: en
edition: Healthcare NLP 3.4.1
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This pretrained pipeline is built on the top of [bert_token_classifier_ner_cellular](https://nlp.johnsnowlabs.com/2022/01/06/bert_token_classifier_ner_cellular_en.html) model.

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/NER_CELLULAR/){:.button.button-orange}
[Open in Colab](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/NER_CELLULAR.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/bert_token_classifier_ner_cellular_pipeline_en_3.4.1_3.0_1646826493144.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/bert_token_classifier_ner_cellular_pipeline_en_3.4.1_3.0_1646826493144.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
from sparknlp.pretrained import PretrainedPipeline

cellular_pipeline = PretrainedPipeline("bert_token_classifier_ner_cellular_pipeline", "en", "clinical/models")

cellular_pipeline.fullAnnotate("Detection of various other intracellular signaling proteins is also described. Genetic characterization of transactivation of the human T-cell leukemia virus type 1 promoter: Binding of Tax to Tax-responsive element 1 is mediated by the cyclic AMP-responsive members of the CREB/ATF family of transcription factors. To achieve a better understanding of the mechanism of transactivation by Tax of human T-cell leukemia virus type 1 Tax-responsive element 1 (TRE-1), we developed a genetic approach with Saccharomyces cerevisiae. We constructed a yeast reporter strain containing the lacZ gene under the control of the CYC1 promoter associated with three copies of TRE-1. Expression of either the cyclic AMP response element-binding protein (CREB) or CREB fused to the GAL4 activation domain (GAD) in this strain did not modify the expression of the reporter gene. Tax alone was also inactive.")
```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val cellular_pipeline = new PretrainedPipeline("bert_token_classifier_ner_cellular_pipeline", "en", "clinical/models")

cellular_pipeline.fullAnnotate("Detection of various other intracellular signaling proteins is also described. Genetic characterization of transactivation of the human T-cell leukemia virus type 1 promoter: Binding of Tax to Tax-responsive element 1 is mediated by the cyclic AMP-responsive members of the CREB/ATF family of transcription factors. To achieve a better understanding of the mechanism of transactivation by Tax of human T-cell leukemia virus type 1 Tax-responsive element 1 (TRE-1), we developed a genetic approach with Saccharomyces cerevisiae. We constructed a yeast reporter strain containing the lacZ gene under the control of the CYC1 promoter associated with three copies of TRE-1. Expression of either the cyclic AMP response element-binding protein (CREB) or CREB fused to the GAL4 activation domain (GAD) in this strain did not modify the expression of the reporter gene. Tax alone was also inactive.")
```
</div>

## Results

```bash
+-----------------------------------------------------------+---------+
|chunk                                                      |ner_label|
+-----------------------------------------------------------+---------+
|intracellular signaling proteins                           |protein  |
|human T-cell leukemia virus type 1 promoter                |DNA      |
|Tax                                                        |protein  |
|Tax-responsive element 1                                   |DNA      |
|cyclic AMP-responsive members                              |protein  |
|CREB/ATF family                                            |protein  |
|transcription factors                                      |protein  |
|Tax                                                        |protein  |
|human T-cell leukemia virus type 1 Tax-responsive element 1|DNA      |
|TRE-1                                                      |DNA      |
|lacZ gene                                                  |DNA      |
|CYC1 promoter                                              |DNA      |
|TRE-1                                                      |DNA      |
|cyclic AMP response element-binding protein                |protein  |
|CREB                                                       |protein  |
|CREB                                                       |protein  |
|GAL4 activation domain                                     |protein  |
|GAD                                                        |protein  |
|reporter gene                                              |DNA      |
|Tax                                                        |protein  |
+-----------------------------------------------------------+---------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_token_classifier_ner_cellular_pipeline|
|Type:|pipeline|
|Compatibility:|Healthcare NLP 3.4.1+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|
|Size:|404.4 MB|

## Included Models

- DocumentAssembler
- TokenizerModel
- MedicalBertForTokenClassifier
- NerConverter
- Finisher
<!--stackedit_data:
eyJoaXN0b3J5IjpbOTk3MjU0MzUxLC0xMjQxMDg5OTddfQ==
-->