---
layout: model
title: Detect Genes/Proteins (BC2GM) in Medical Texts
author: John Snow Labs
name: ner_biomedical_bc2gm_pipeline
date: 2022-06-22
tags: [licensed, clinical, en, ner, bc2gm, gene_protein, gene, protein, biomedical]
task: Pipeline Healthcare
language: en
edition: Healthcare NLP 3.5.3
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This pretrained pipeline is built on the top of [ner_biomedical_bc2gm](https://nlp.johnsnowlabs.com/2022/05/10/ner_biomedical_bc2gm_en_3_0.html) model.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/11.Pretrained_Clinical_Pipelines.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_biomedical_bc2gm_pipeline_en_3.5.3_3.0_1655893015210.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/ner_biomedical_bc2gm_pipeline_en_3.5.3_3.0_1655893015210.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline

pipeline = PretrainedPipeline("ner_biomedical_bc2gm_pipeline", "en", "clinical/models")

result = pipeline.fullAnnotate("""Immunohistochemical staining was positive for S-100 in all 9 cases stained, positive for HMB-45 in 9 (90%) of 10, and negative for cytokeratin in all 9 cases in which myxoid melanoma remained in the block after previous sections.""")
```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val pipeline = new PretrainedPipeline("ner_biomedical_bc2gm_pipeline", "en", "clinical/models")

val result = pipeline.fullAnnotate("""Immunohistochemical staining was positive for S-100 in all 9 cases stained, positive for HMB-45 in 9 (90%) of 10, and negative for cytokeratin in all 9 cases in which myxoid melanoma remained in the block after previous sections""")
```


{:.nlu-block}
```python
import nlu
nlu.load("en.med_ner.biomedical_bc2gm.pipeline").predict("""Immunohistochemical staining was positive for S-100 in all 9 cases stained, positive for HMB-45 in 9 (90%) of 10, and negative for cytokeratin in all 9 cases in which myxoid melanoma remained in the block after previous sections.""")
```

</div>

## Results

```bash
+-----------+------------+
|chunk      |ner_label   |
+-----------+------------+
|S-100      |GENE_PROTEIN|
|HMB-45     |GENE_PROTEIN|
|cytokeratin|GENE_PROTEIN|
+-----------+------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_biomedical_bc2gm_pipeline|
|Type:|pipeline|
|Compatibility:|Healthcare NLP 3.5.3+|
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
- NerConverter
