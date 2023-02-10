---
layout: model
title: Recognize Entities Bert Noncontrib
author: John Snow Labs
name: recognize_entities_bert_noncontrib
date: 2022-06-20
tags: [en, open_source]
task: Named Entity Recognition
language: en
edition: Spark NLP 4.0.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

The recognize_entities_bert_noncontrib is a pretrained pipeline that we can use to process text with a simple pipeline that performs basic processing steps and recognizes entites.
         It performs most of the common text processing tasks on your dataframe

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/recognize_entities_bert_noncontrib_en_4.0.0_3.0_1655697462434.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/recognize_entities_bert_noncontrib_en_4.0.0_3.0_1655697462434.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline
pipeline = PretrainedPipeline("recognize_entities_bert_noncontrib", "en", "clinical/models")

result = pipeline.annotate("""I love johnsnowlabs!  """)
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|recognize_entities_bert_noncontrib|
|Type:|pipeline|
|Compatibility:|Spark NLP 4.0.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|425.6 MB|

## Included Models

- DocumentAssembler
- SentenceDetector
- TokenizerModel
- BertEmbeddings
- NerDLModel