---
layout: model
title: Pipeline to Detect bacterial species
author: John Snow Labs
name: ner_bacterial_species_pipeline
date: 2023-03-14
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

This pretrained pipeline is built on the top of [ner_bacterial_species](https://nlp.johnsnowlabs.com/2021/04/01/ner_bacterial_species_en.html) model.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_bacterial_species_pipeline_en_4.3.0_3.2_1678804215844.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/ner_bacterial_species_pipeline_en_4.3.0_3.2_1678804215844.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline

pipeline = PretrainedPipeline("ner_bacterial_species_pipeline", "en", "clinical/models")

text = '''Based on these genetic and phenotypic properties, we propose that strain SMSP (T) represents a novel species of the genus Methanoregula, for which we propose the name Methanoregula formicica sp. nov., with the type strain SMSP (T) (= NBRC 105244 (T) = DSM 22288 (T)).'''

result = pipeline.fullAnnotate(text)
```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val pipeline = new PretrainedPipeline("ner_bacterial_species_pipeline", "en", "clinical/models")

val text = "Based on these genetic and phenotypic properties, we propose that strain SMSP (T) represents a novel species of the genus Methanoregula, for which we propose the name Methanoregula formicica sp. nov., with the type strain SMSP (T) (= NBRC 105244 (T) = DSM 22288 (T))."

val result = pipeline.fullAnnotate(text)
```
</div>

## Results

```bash
|    | ner_chunks              |   begin |   end | ner_label   |   confidence |
|---:|:------------------------|--------:|------:|:------------|-------------:|
|  0 | SMSP (T)                |      73 |    80 | SPECIES     |     0.9725   |
|  1 | Methanoregula formicica |     167 |   189 | SPECIES     |     0.97935  |
|  2 | SMSP (T)                |     222 |   229 | SPECIES     |     0.991975 |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_bacterial_species_pipeline|
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