---
layout: model
title: Icelandic NER Pipeline
author: John Snow Labs
name: roberta_token_classifier_icelandic_ner_pipeline
date: 2023-05-21
tags: [open_source, ner, token_classifier, roberta, icelandic, is]
task: Named Entity Recognition
language: is
edition: Spark NLP 4.4.2
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This pretrained pipeline is built on the top of [roberta_token_classifier_icelandic_ner](https://nlp.johnsnowlabs.com/2021/12/06/roberta_token_classifier_icelandic_ner_is.html) model.

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/roberta_token_classifier_icelandic_ner_pipeline_is_4.4.2_3.0_1684652336165.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/roberta_token_classifier_icelandic_ner_pipeline_is_4.4.2_3.0_1684652336165.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("roberta_token_classifier_icelandic_ner_pipeline", lang = "is")

pipeline.annotate("Ég heiti Peter Fergusson. Ég hef búið í New York síðan í október 2011 og unnið hjá Tesla Motor og þénað 100K $ á ári.")
```
```scala

val pipeline = new PretrainedPipeline("roberta_token_classifier_icelandic_ner_pipeline", lang = "is")

pipeline.annotate("Ég heiti Peter Fergusson. Ég hef búið í New York síðan í október 2011 og unnið hjá Tesla Motor og þénað 100K $ á ári.")
```
</div>

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
pipeline = PretrainedPipeline("roberta_token_classifier_icelandic_ner_pipeline", lang = "is")

pipeline.annotate("Ég heiti Peter Fergusson. Ég hef búið í New York síðan í október 2011 og unnið hjá Tesla Motor og þénað 100K $ á ári.")
```
```scala
val pipeline = new PretrainedPipeline("roberta_token_classifier_icelandic_ner_pipeline", lang = "is")

pipeline.annotate("Ég heiti Peter Fergusson. Ég hef búið í New York síðan í október 2011 og unnið hjá Tesla Motor og þénað 100K $ á ári.")
```
</div>

## Results

```bash
Results



+----------------+------------+
|chunk           |ner_label   |
+----------------+------------+
|Peter Fergusson |Person      |
|New York        |Location    |
|október 2011    |Date        |
|Tesla Motor     |Organization|
|100K $          |Money       |
+----------------+------------+


{:.model-param}
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|roberta_token_classifier_icelandic_ner_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 4.4.2+|
|License:|Open Source|
|Edition:|Official|
|Language:|is|
|Size:|457.6 MB|

## Included Models

- DocumentAssembler
- SentenceDetectorDLModel
- TokenizerModel
- RoBertaForTokenClassification
- NerConverter
- Finisher