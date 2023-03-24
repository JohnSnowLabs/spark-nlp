---
layout: model
title: Pipeline to Detect Anatomical and Observation Entities in Chest Radiology Reports (CheXpert)
author: John Snow Labs
name: ner_chexpert_pipeline
date: 2023-03-14
tags: [licensed, ner, clinical, en]
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

This pretrained pipeline is built on the top of [ner_chexpert](https://nlp.johnsnowlabs.com/2021/09/30/ner_chexpert_en.html) model.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_chexpert_pipeline_en_4.3.0_3.2_1678779791404.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/ner_chexpert_pipeline_en_4.3.0_3.2_1678779791404.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline

pipeline = PretrainedPipeline("ner_chexpert_pipeline", "en", "clinical/models")

text = '''FINAL REPORT HISTORY : Chest tube leak , to assess for pneumothorax. FINDINGS : In comparison with study of ___ , the endotracheal tube and Swan - Ganz catheter have been removed . The left chest tube remains in place and there is no evidence of pneumothorax. Mild atelectatic changes are seen at the left base.'''

result = pipeline.fullAnnotate(text)
```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val pipeline = new PretrainedPipeline("ner_chexpert_pipeline", "en", "clinical/models")

val text = "FINAL REPORT HISTORY : Chest tube leak , to assess for pneumothorax. FINDINGS : In comparison with study of ___ , the endotracheal tube and Swan - Ganz catheter have been removed . The left chest tube remains in place and there is no evidence of pneumothorax. Mild atelectatic changes are seen at the left base."

val result = pipeline.fullAnnotate(text)
```
</div>

## Results

```bash
|    | ner_chunks   |   begin |   end | ner_label   |   confidence |
|---:|:-------------|--------:|------:|:------------|-------------:|
|  0 | endotracheal |     118 |   129 | OBS         |       0.9881 |
|  1 | tube         |     131 |   134 | OBS         |       0.9996 |
|  2 | Swan - Ganz  |     140 |   150 | OBS         |       0.9625 |
|  3 | catheter     |     152 |   159 | OBS         |       0.9919 |
|  4 | left         |     185 |   188 | ANAT        |       0.9983 |
|  5 | chest        |     190 |   194 | ANAT        |       0.9749 |
|  6 | tube         |     196 |   199 | OBS         |       0.9999 |
|  7 | in place     |     209 |   216 | OBS         |       0.9894 |
|  8 | pneumothorax |     246 |   257 | OBS         |       0.9997 |
|  9 | Mild         |     260 |   263 | OBS         |       0.9988 |
| 10 | atelectatic  |     265 |   275 | OBS         |       0.9986 |
| 11 | changes      |     277 |   283 | OBS         |       0.9984 |
| 12 | left         |     301 |   304 | ANAT        |       0.9999 |
| 13 | base         |     306 |   309 | ANAT        |       0.9999 |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_chexpert_pipeline|
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