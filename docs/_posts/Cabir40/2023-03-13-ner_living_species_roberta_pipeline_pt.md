---
layout: model
title: Pipeline to Detect Living Species(roberta_embeddings_BR_BERTo)
author: John Snow Labs
name: ner_living_species_roberta_pipeline
date: 2023-03-13
tags: [pt, ner, clinical, licensed, roberta]
task: Named Entity Recognition
language: pt
edition: Healthcare NLP 4.3.0
spark_version: 3.2
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This pretrained pipeline is built on the top of [ner_living_species_roberta](https://nlp.johnsnowlabs.com/2022/06/22/ner_living_species_roberta_pt_3_0.html) model.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_living_species_roberta_pipeline_pt_4.3.0_3.2_1678732150750.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/ner_living_species_roberta_pipeline_pt_4.3.0_3.2_1678732150750.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline

pipeline = PretrainedPipeline("ner_living_species_roberta_pipeline", "pt", "clinical/models")

text = '''Mulher de 23 anos, de Capinota, Cochabamba, Bolívia. Ela está no nosso país há quatro anos. Frequentou o departamento de emergência obstétrica onde foi encontrada grávida de 37 semanas, com um colo dilatado de 5 cm e membranas rompidas. O obstetra de emergência realizou um teste de estreptococos negativo e solicitou um hemograma, glucose, bioquímica básica, HBV, HCV e serologia da sífilis.'''

result = pipeline.fullAnnotate(text)
```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val pipeline = new PretrainedPipeline("ner_living_species_roberta_pipeline", "pt", "clinical/models")

val text = "Mulher de 23 anos, de Capinota, Cochabamba, Bolívia. Ela está no nosso país há quatro anos. Frequentou o departamento de emergência obstétrica onde foi encontrada grávida de 37 semanas, com um colo dilatado de 5 cm e membranas rompidas. O obstetra de emergência realizou um teste de estreptococos negativo e solicitou um hemograma, glucose, bioquímica básica, HBV, HCV e serologia da sífilis."

val result = pipeline.fullAnnotate(text)
```
</div>

## Results

```bash
|    | ner_chunks    |   begin |   end | ner_label   |   confidence |
|---:|:--------------|--------:|------:|:------------|-------------:|
|  0 | Mulher        |       0 |     5 | HUMAN       |       0.9975 |
|  1 | país          |      71 |    74 | HUMAN       |       0.8869 |
|  2 | grávida       |     163 |   169 | HUMAN       |       0.9702 |
|  3 | estreptococos |     283 |   295 | SPECIES     |       0.9211 |
|  4 | HBV           |     360 |   362 | SPECIES     |       0.9911 |
|  5 | HCV           |     365 |   367 | SPECIES     |       0.9858 |
|  6 | sífilis       |     384 |   390 | SPECIES     |       0.8898 |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_living_species_roberta_pipeline|
|Type:|pipeline|
|Compatibility:|Healthcare NLP 4.3.0+|
|License:|Licensed|
|Edition:|Official|
|Language:|pt|
|Size:|654.0 MB|

## Included Models

- DocumentAssembler
- SentenceDetectorDLModel
- TokenizerModel
- RoBertaEmbeddings
- MedicalNerModel
- NerConverterInternalModel