---
layout: model
title: Pipeline to Detect details of cellular structures (biobert)
author: John Snow Labs
name: ner_cellular_biobert_pipeline
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

This pretrained pipeline is built on the top of [ner_cellular_biobert](https://nlp.johnsnowlabs.com/2021/04/01/ner_cellular_biobert_en.html) model.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_cellular_biobert_pipeline_en_4.3.0_3.2_1679314449983.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/ner_cellular_biobert_pipeline_en_4.3.0_3.2_1679314449983.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline

pipeline = PretrainedPipeline("ner_cellular_biobert_pipeline", "en", "clinical/models")

text = '''Detection of various other intracellular signaling proteins is also described. Genetic characterization of transactivation of the human T-cell leukemia virus type 1 promoter: Binding of Tax to Tax-responsive element 1 is mediated by the cyclic AMP-responsive members of the CREB/ATF family of transcription factors. To achieve a better understanding of the mechanism of transactivation by Tax of human T-cell leukemia virus type 1 Tax-responsive element 1 (TRE-1), we developed a genetic approach with Saccharomyces cerevisiae. We constructed a yeast reporter strain containing the lacZ gene under the control of the CYC1 promoter associated with three copies of TRE-1. Expression of either the cyclic AMP response element-binding protein (CREB) or CREB fused to the GAL4 activation domain (GAD) in this strain did not modify the expression of the reporter gene. Tax alone was also inactive.'''

result = pipeline.fullAnnotate(text)
```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val pipeline = new PretrainedPipeline("ner_cellular_biobert_pipeline", "en", "clinical/models")

val text = "Detection of various other intracellular signaling proteins is also described. Genetic characterization of transactivation of the human T-cell leukemia virus type 1 promoter: Binding of Tax to Tax-responsive element 1 is mediated by the cyclic AMP-responsive members of the CREB/ATF family of transcription factors. To achieve a better understanding of the mechanism of transactivation by Tax of human T-cell leukemia virus type 1 Tax-responsive element 1 (TRE-1), we developed a genetic approach with Saccharomyces cerevisiae. We constructed a yeast reporter strain containing the lacZ gene under the control of the CYC1 promoter associated with three copies of TRE-1. Expression of either the cyclic AMP response element-binding protein (CREB) or CREB fused to the GAL4 activation domain (GAD) in this strain did not modify the expression of the reporter gene. Tax alone was also inactive."

val result = pipeline.fullAnnotate(text)
```
</div>

## Results

```bash
|    | ner_chunk                                   |   begin |   end | ner_label   |   confidence |
|---:|:--------------------------------------------|--------:|------:|:------------|-------------:|
|  0 | intracellular signaling proteins            |      27 |    58 | protein     |     0.673333 |
|  1 | human T-cell leukemia virus type 1 promoter |     130 |   172 | DNA         |     0.426171 |
|  2 | Tax                                         |     186 |   188 | protein     |     0.779    |
|  3 | Tax-responsive element 1                    |     193 |   216 | DNA         |     0.756933 |
|  4 | cyclic AMP-responsive members               |     237 |   265 | protein     |     0.629333 |
|  5 | CREB/ATF family                             |     274 |   288 | protein     |     0.8499   |
|  6 | transcription factors                       |     293 |   313 | protein     |     0.78165  |
|  7 | Tax                                         |     389 |   391 | protein     |     0.8463   |
|  8 | Tax-responsive element 1                    |     431 |   454 | DNA         |     0.713067 |
|  9 | TRE-1                                       |     457 |   461 | DNA         |     0.9983   |
| 10 | lacZ gene                                   |     582 |   590 | DNA         |     0.7018   |
| 11 | CYC1 promoter                               |     617 |   629 | DNA         |     0.81865  |
| 12 | TRE-1                                       |     663 |   667 | DNA         |     0.9967   |
| 13 | cyclic AMP response element-binding protein |     695 |   737 | protein     |     0.51984  |
| 14 | CREB                                        |     740 |   743 | protein     |     0.9708   |
| 15 | CREB                                        |     749 |   752 | protein     |     0.8875   |
| 16 | GAL4 activation domain                      |     767 |   788 | protein     |     0.578633 |
| 17 | GAD                                         |     791 |   793 | protein     |     0.6432   |
| 18 | reporter gene                               |     848 |   860 | DNA         |     0.61005  |
| 19 | Tax                                         |     863 |   865 | protein     |     0.99     |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_cellular_biobert_pipeline|
|Type:|pipeline|
|Compatibility:|Healthcare NLP 4.3.0+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|
|Size:|422.1 MB|

## Included Models

- DocumentAssembler
- SentenceDetectorDLModel
- TokenizerModel
- BertEmbeddings
- MedicalNerModel
- NerConverterInternalModel