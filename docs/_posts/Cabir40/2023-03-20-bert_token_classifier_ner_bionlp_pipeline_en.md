---
layout: model
title: Pipeline to Detect Cancer Genetics (BertForTokenClassification)
author: John Snow Labs
name: bert_token_classifier_ner_bionlp_pipeline
date: 2023-03-20
tags: [bertfortokenclassification, ner, bionlp, en, licensed]
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

This pretrained pipeline is built on the top of [bert_token_classifier_ner_bionlp](https://nlp.johnsnowlabs.com/2022/01/03/bert_token_classifier_ner_bionlp_en.html) model.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/bert_token_classifier_ner_bionlp_pipeline_en_4.3.0_3.2_1679308593451.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/bert_token_classifier_ner_bionlp_pipeline_en_4.3.0_3.2_1679308593451.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline

pipeline = PretrainedPipeline("bert_token_classifier_ner_bionlp_pipeline", "en", "clinical/models")

text = '''Both the erbA IRES and the erbA/myb virus constructs transformed erythroid cells after infection of bone marrow or blastoderm cultures. The erbA/myb IRES virus exhibited a 5-10-fold higher transformed colony forming efficiency than the erbA IRES virus in the blastoderm assay.'''

result = pipeline.fullAnnotate(text)
```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val pipeline = new PretrainedPipeline("bert_token_classifier_ner_bionlp_pipeline", "en", "clinical/models")

val text = "Both the erbA IRES and the erbA/myb virus constructs transformed erythroid cells after infection of bone marrow or blastoderm cultures. The erbA/myb IRES virus exhibited a 5-10-fold higher transformed colony forming efficiency than the erbA IRES virus in the blastoderm assay."

val result = pipeline.fullAnnotate(text)
```
</div>

## Results

```bash
|    | ner_chunk           |   begin |   end | ner_label              |   confidence |
|---:|:--------------------|--------:|------:|:-----------------------|-------------:|
|  0 | erbA IRES           |       9 |    17 | Organism               |     0.999188 |
|  1 | erbA/myb virus      |      27 |    40 | Organism               |     0.999434 |
|  2 | erythroid cells     |      65 |    79 | Cell                   |     0.999837 |
|  3 | bone                |     100 |   103 | Multi-tissue_structure |     0.999846 |
|  4 | marrow              |     105 |   110 | Multi-tissue_structure |     0.999876 |
|  5 | blastoderm cultures |     115 |   133 | Cell                   |     0.999823 |
|  6 | erbA/myb IRES virus |     140 |   158 | Organism               |     0.999751 |
|  7 | erbA IRES virus     |     236 |   250 | Organism               |     0.999749 |
|  8 | blastoderm          |     259 |   268 | Cell                   |     0.999897 |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_token_classifier_ner_bionlp_pipeline|
|Type:|pipeline|
|Compatibility:|Healthcare NLP 4.3.0+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|
|Size:|405.0 MB|

## Included Models

- DocumentAssembler
- SentenceDetectorDLModel
- TokenizerModel
- MedicalBertForTokenClassifier
- NerConverterInternalModel