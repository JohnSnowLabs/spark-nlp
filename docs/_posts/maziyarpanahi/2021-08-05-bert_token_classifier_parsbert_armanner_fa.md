---
layout: model
title: BERT Token Classification - ParsBERT for Persian Language Understanding (bert_token_classifier_parsbert_armanner)
author: John Snow Labs
name: bert_token_classifier_parsbert_armanner
date: 2021-08-05
tags: [fa, persian, farsi, ner, token_classification, bert, parsbert, open_source]
task: Named Entity Recognition
language: fa
edition: Spark NLP 3.2.0
spark_version: 2.4
supported: true
annotator: BertForTokenClassification
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

## ParsBERT: Transformer-based Model for Persian Language Understanding

ParsBERT is a monolingual language model based on Google’s BERT architecture with the same configurations as BERT-Base. 

Paper presenting ParsBERT: [arXiv:2005.12515](https://arxiv.org/abs/2005.12515)

All the models (downstream tasks) are uncased and trained with whole word masking. (coming soon stay tuned)

## Persian NER [ARMAN, PEYMA, ARMAN+PEYMA]

This task aims to extract named entities in the text, such as names, and label them with appropriate `NER` classes such as locations, organizations, etc. The datasets used for this task contain sentences that are marked with `IOB` format. In this format, tokens that are not part of an entity are tagged as `”O”` the `”B”`tag corresponds to the first word of an object, and the `”I”` tag corresponds to the rest of the terms of the same entity. Both `”B”` and `”I”` tags are followed by a hyphen (or underscore), followed by the entity category. Therefore, the NER task is a multi-class token classification problem that labels the tokens upon being fed a raw text. There are two primary datasets used in Persian NER, `ARMAN`, and `PEYMA`. In ParsBERT, we prepared ner for both datasets as well as a combination of both datasets.

### ARMAN

ARMAN dataset holds 7,682 sentences with 250,015 sentences tagged over six different classes.
1. Organization
2. Location
3. Facility
4. Event
5. Product
6. Person

|     Label    |   #   |
|:------------:|:-----:|
| Organization | 30108 |
|   Location   | 12924 |
|   Facility   |  4458 |
|     Event    |  7557 |
|    Product   |  4389 |
|    Person    | 15645 |

## Cite 
Please cite the following paper in your publication if you are using [ParsBERT](https://arxiv.org/abs/2005.12515) in your research:
```markdown
@article{ParsBERT,
title={ParsBERT: Transformer-based Model for Persian Language Understanding},
author={Mehrdad Farahani, Mohammad Gharachorloo, Marzieh Farahani, Mohammad Manthouri},
journal={ArXiv},
year={2020},
volume={abs/2005.12515}
}
```

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_token_classifier_parsbert_armanner_fa_3.2.0_2.4_1628181836965.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
document_assembler = DocumentAssembler() \
.setInputCol('text') \
.setOutputCol('document')

tokenizer = Tokenizer() \
.setInputCols(['document']) \
.setOutputCol('token')

tokenClassifier = BertForTokenClassification \
.pretrained('bert_token_classifier_parsbert_armanner', 'fa') \
.setInputCols(['token', 'document']) \
.setOutputCol('ner') \
.setCaseSensitive(False) \
.setMaxSentenceLength(512)

# since output column is IOB/IOB2 style, NerConverter can extract entities
ner_converter = NerConverter() \
.setInputCols(['document', 'token', 'ner']) \
.setOutputCol('entities')

pipeline = Pipeline(stages=[
document_assembler, 
tokenizer,
tokenClassifier,
ner_converter
])

example = spark.createDataFrame([["دفتر مرکزی شرکت کامیکو در شهر ساسکاتون ساسکاچوان قرار دارد."]]).toDF("text")
result = pipeline.fit(example).transform(example)
```
```scala
val document_assembler = DocumentAssembler() 
.setInputCol("text") 
.setOutputCol("document")

val tokenizer = Tokenizer() 
.setInputCols("document") 
.setOutputCol("token")

val tokenClassifier = BertForTokenClassification.pretrained("bert_token_classifier_parsbert_armanner", "fa")
.setInputCols("document", "token")
.setOutputCol("ner")
.setCaseSensitive(false)
.setMaxSentenceLength(512)

// since output column is IOB/IOB2 style, NerConverter can extract entities
val ner_converter = NerConverter() 
.setInputCols("document", "token", "ner") 
.setOutputCol("entities")

val pipeline = new Pipeline().setStages(Array(document_assembler, tokenizer, tokenClassifier, ner_converter))

val example = Seq.empty["دفتر مرکزی شرکت کامیکو در شهر ساسکاتون ساسکاچوان قرار دارد."].toDS.toDF("text")

val result = pipeline.fit(example).transform(example)
```


{:.nlu-block}
```python
import nlu
nlu.load("fa.classify.token_bert.parsbert_armanner").predict("""دفتر مرکزی شرکت کامیکو در شهر ساسکاتون ساسکاچوان قرار دارد.""")
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_token_classifier_parsbert_armanner|
|Compatibility:|Spark NLP 3.2.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[token, document]|
|Output Labels:|[ner]|
|Language:|fa|
|Case sensitive:|true|
|Max sentense length:|512|

## Data Source

[https://huggingface.co/HooshvareLab/bert-base-parsbert-armanner-uncased](https://huggingface.co/HooshvareLab/bert-base-parsbert-armanner-uncased)

## Benchmarking

```bash
The following table summarizes the F1 score obtained by ParsBERT as compared to other models and architectures.

| Dataset | ParsBERT | MorphoBERT | Beheshti-NER | LSTM-CRF | Rule-Based CRF | BiLSTM-CRF |
|---------|----------|------------|--------------|----------|----------------|------------|
| ARMAN   | 93.10*   | 89.9       | 84.03        | 86.55    | -              | 77.45      |

```
