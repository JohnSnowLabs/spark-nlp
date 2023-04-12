---
layout: model
title: Recognize Entities OntoNotes - BERT Medium
author: John Snow Labs
name: onto_recognize_entities_bert_medium
date: 2020-12-09
task: [Named Entity Recognition, Sentence Detection, Embeddings, Pipeline Public]
language: en
nav_key: models
edition: Spark NLP 2.7.0
spark_version: 2.4
tags: [open_source, en, pipeline]
supported: true
annotator: PipelineModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

A pre-trained pipeline containing NerDl Model. The NER model trained on OntoNotes 5.0 with `small_bert_L8_512` embeddings. It can extract up to following 18 entities:

## Predicted Entities
`CARDINAL`, `DATE`, `EVENT`, `FAC`, `GPE`, `LANGUAGE`, `LAW`, `LOC`, `MONEY`, `NORP`, `ORDINAL`, `ORG`, `PERCENT`, `PERSON`, `PRODUCT`, `QUANTITY`, `TIME`, `WORK_OF_ART`.

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/public/NER_EN_18/){:.button.button-orange}
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/onto_recognize_entities_bert_medium_en_2.7.0_2.4_1607510751761.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/onto_recognize_entities_bert_medium_en_2.7.0_2.4_1607510751761.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
from sparknlp.pretrained import PretrainedPipeline

pipeline = PretrainedPipeline('onto_recognize_entities_bert_medium')

result = pipeline.annotate("Johnson first entered politics when elected in 2001 as a member of Parliament. He then served eight years as the mayor of London, from 2008 to 2016, before rejoining Parliament.")
```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val pipeline = new PretrainedPipeline("onto_recognize_entities_bert_medium")

val result = pipeline.annotate("Johnson first entered politics when elected in 2001 as a member of Parliament. He then served eight years as the mayor of London, from 2008 to 2016, before rejoining Parliament.")
```

{:.nlu-block}
```python
import nlu

text = ["""Johnson first entered politics when elected in 2001 as a member of Parliament. He then served eight years as the mayor of London, from 2008 to 2016, before rejoining Parliament."""]
ner_df = nlu.load('en.ner.onto.bert.medium').predict(text, output_level='chunk')
ner_df[["entities", "entities_class"]]
```

</div>

{:.h2_title}
## Results

```bash
+------------+---------+
|chunk       |ner_label|
+------------+---------+
|Johnson     |PERSON   |
|first       |ORDINAL  |
|2001        |DATE     |
|eight years |DATE     |
|London      |GPE      |
|2008 to 2016|DATE     |
+------------+---------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|onto_recognize_entities_bert_medium|
|Type:|pipeline|
|Compatibility:|Spark NLP 2.7.0+|
|Edition:|Official|
|Language:|en|

## Included Models

- DocumentAssembler
- SentenceDetectorDLModel
- Tokenizer
- BertEmbeddings
- NerDLModel
- NerConverter