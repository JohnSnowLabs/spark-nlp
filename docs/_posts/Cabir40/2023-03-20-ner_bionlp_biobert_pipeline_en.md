---
layout: model
title: Pipeline to Detect biological concepts (biobert)
author: John Snow Labs
name: ner_bionlp_biobert_pipeline
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

This pretrained pipeline is built on the top of [ner_bionlp_biobert](https://nlp.johnsnowlabs.com/2021/04/01/ner_bionlp_biobert_en.html) model.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_bionlp_biobert_pipeline_en_4.3.0_3.2_1679313010526.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/ner_bionlp_biobert_pipeline_en_4.3.0_3.2_1679313010526.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline

pipeline = PretrainedPipeline("ner_bionlp_biobert_pipeline", "en", "clinical/models")

text = '''Both the erbA IRES and the erbA/myb virus constructs transformed erythroid cells after infection of bone marrow or blastoderm cultures. The erbA/myb IRES virus exhibited a 5-10-fold higher transformed colony forming efficiency than the erbA IRES virus in the blastoderm assay'''

result = pipeline.fullAnnotate(text)
```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val pipeline = new PretrainedPipeline("ner_bionlp_biobert_pipeline", "en", "clinical/models")

val text = "Both the erbA IRES and the erbA/myb virus constructs transformed erythroid cells after infection of bone marrow or blastoderm cultures. The erbA/myb IRES virus exhibited a 5-10-fold higher transformed colony forming efficiency than the erbA IRES virus in the blastoderm assay"

val result = pipeline.fullAnnotate(text)
```
</div>

## Results

```bash
|    | ner_chunk           |   begin |   end | ner_label              |   confidence |
|---:|:--------------------|--------:|------:|:-----------------------|-------------:|
|  0 | erbA                |       9 |    12 | Gene_or_gene_product   |      1       |
|  1 | IRES                |      14 |    17 | Organism               |      0.754   |
|  2 | virus               |      36 |    40 | Organism               |      0.9999  |
|  3 | erythroid cells     |      65 |    79 | Cell                   |      0.99855 |
|  4 | bone                |     100 |   103 | Multi-tissue_structure |      0.9794  |
|  5 | marrow              |     105 |   110 | Multi-tissue_structure |      0.9631  |
|  6 | blastoderm cultures |     115 |   133 | Cell                   |      0.9868  |
|  7 | IRES virus          |     149 |   158 | Organism               |      0.99985 |
|  8 | erbA                |     236 |   239 | Gene_or_gene_product   |      0.9977  |
|  9 | IRES virus          |     241 |   250 | Organism               |      0.9911  |
| 10 | blastoderm          |     259 |   268 | Cell                   |      0.9941  |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_bionlp_biobert_pipeline|
|Type:|pipeline|
|Compatibility:|Healthcare NLP 4.3.0+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|
|Size:|422.2 MB|

## Included Models

- DocumentAssembler
- SentenceDetectorDLModel
- TokenizerModel
- BertEmbeddings
- MedicalNerModel
- NerConverterInternalModel