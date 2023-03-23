---
layout: model
title: Pipeline to Detect Chemicals in Medical Text
author: John Snow Labs
name: bert_token_classifier_ner_bc5cdr_chemicals_pipeline
date: 2023-03-20
tags: [en, ner, clinical, licensed, bertfortokenclasification]
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

This pretrained pipeline is built on the top of [bert_token_classifier_ner_bc5cdr_chemicals](https://nlp.johnsnowlabs.com/2022/07/25/bert_token_classifier_ner_bc5cdr_chemicals_en_3_0.html) model.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/bert_token_classifier_ner_bc5cdr_chemicals_pipeline_en_4.3.0_3.2_1679301940550.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/bert_token_classifier_ner_bc5cdr_chemicals_pipeline_en_4.3.0_3.2_1679301940550.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline

pipeline = PretrainedPipeline("bert_token_classifier_ner_bc5cdr_chemicals_pipeline", "en", "clinical/models")

text = '''The possibilities that these cardiovascular findings might be the result of non-selective inhibition of monoamine oxidase or of amphetamine and metamphetamine are discussed. The results have shown that the degradation product p-choloroaniline is not a significant factor in chlorhexidine-digluconate associated erosive cystitis. A high percentage of kanamycin - colistin and povidone-iodine irrigations were associated with erosive cystitis and suggested a possible complication with human usage.'''

result = pipeline.fullAnnotate(text)
```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val pipeline = new PretrainedPipeline("bert_token_classifier_ner_bc5cdr_chemicals_pipeline", "en", "clinical/models")

val text = "The possibilities that these cardiovascular findings might be the result of non-selective inhibition of monoamine oxidase or of amphetamine and metamphetamine are discussed. The results have shown that the degradation product p-choloroaniline is not a significant factor in chlorhexidine-digluconate associated erosive cystitis. A high percentage of kanamycin - colistin and povidone-iodine irrigations were associated with erosive cystitis and suggested a possible complication with human usage."

val result = pipeline.fullAnnotate(text)
```
</div>

## Results

```bash
|    | ner_chunk                 |   begin |   end | ner_label   |   confidence |
|---:|:--------------------------|--------:|------:|:------------|-------------:|
|  0 | amphetamine               |     128 |   138 | CHEM        |     0.999973 |
|  1 | metamphetamine            |     144 |   157 | CHEM        |     0.999972 |
|  2 | p-choloroaniline          |     226 |   241 | CHEM        |     0.588953 |
|  3 | chlorhexidine-digluconate |     274 |   298 | CHEM        |     0.999979 |
|  4 | kanamycin                 |     350 |   358 | CHEM        |     0.999978 |
|  5 | colistin                  |     362 |   369 | CHEM        |     0.999942 |
|  6 | povidone-iodine           |     375 |   389 | CHEM        |     0.999977 |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_token_classifier_ner_bc5cdr_chemicals_pipeline|
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