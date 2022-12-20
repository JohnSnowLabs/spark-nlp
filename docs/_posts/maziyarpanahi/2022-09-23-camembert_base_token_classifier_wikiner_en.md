---
layout: model
title: CamemBERT Token Classification Base - French WikiNER (camembert_base_token_classifier_wikiner)
author: John Snow Labs
name: camembert_base_token_classifier_wikiner
date: 2022-09-23
tags: [open_source, camembert, token_classification, wikiner, ner, fr, french]
task: Named Entity Recognition
language: en
edition: Spark NLP 4.2.0
spark_version: 3.0
supported: true
annotator: CamemBertForTokenClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

CamemBertForTokenClassification can load CamemBERT Models with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for Named-Entity-Recognition (NER) tasks.

camembert_base_token_classifier_wikiner is a fine-tuned CamemBERT model that is ready to be used for Token Classification task such as Named Entity Recognition and it achieves state-of-the-art performance.

We used TFCamembertForTokenClassification to train this model and used CamemBertForTokenClassification annotator in Spark NLP 🚀 for prediction at scale!  This model has been trained to recognize four types of entities: 
O, LOC, PER, MISC, and ORG.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/camembert_base_token_classifier_wikiner_en_4.2.0_3.0_1663928016186.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
                
document_assembler = DocumentAssembler()\ 
    .setInputCol("text")\ 
    .setOutputCol("document")

tokenizer = Tokenizer()\ 
    .setInputCols(['document'])\ 
    .setOutputCol('token') 

tokenClassifier = CamemBertForTokenClassification.pretrained("camembert_base_token_classifier_wikiner", "en")\ 
    .setInputCols(["document", "token"])\ 
    .setOutputCol("ner")\ 
    .setCaseSensitive(True)\ 
    .setMaxSentenceLength(512) 

# since output column is IOB/IOB2 style, NerConverter can extract entities
ner_converter = NerConverter()\ 
    .setInputCols(['document', 'token', 'ner'])\ 
    .setOutputCol('entities') 

pipeline = Pipeline(stages=[
    document_assembler,
    tokenizer,
    tokenClassifier,
    ner_converter
])

example = spark.createDataFrame([['I really liked that movie!']]).toDF("text")
result = pipeline.fit(example).transform(example)
```
```scala

val document_assembler = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val tokenizer = new Tokenizer()
    .setInputCols("document")
    .setOutputCol("token")

val tokenClassifier = CamemBertForTokenClassification.pretrained("camembert_base_token_classifier_wikiner", "en")
    .setInputCols("document", "token")
    .setOutputCol("ner")
    .setCaseSensitive(true)
    .setMaxSentenceLength(512)

// since output column is IOB/IOB2 style, NerConverter can extract entities
val ner_converter = NerConverter() 
    .setInputCols("document", "token", "ner") 
    .setOutputCol("entities")

val pipeline = new Pipeline().setStages(Array(document_assembler, tokenizer, tokenClassifier, ner_converter))

val example = Seq("I really liked that movie!").toDS.toDF("text")

val result = pipeline.fit(example).transform(example)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|camembert_base_token_classifier_wikiner|
|Compatibility:|Spark NLP 4.2.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[token, document]|
|Output Labels:|[ner]|
|Language:|en|
|Size:|412.2 MB|
|Case sensitive:|true|
|Max sentence length:|512|

## References

[https://huggingface.co/datasets/Jean-Baptiste/wikiner_fr](https://huggingface.co/datasets/Jean-Baptiste/wikiner_fr)

## Benchmarking

```bash
root: "{"
eval_loss ": 0.0396629199385643, "
eval_precision ": 0.9283267457180501, "
eval_recall ": 0.9337809780447939, "
eval_f1 ": 0.9310458739841875, "
eval_accuracy ": 0.9896610647939648, "
eval_runtime ": 204.7064, "
eval_samples_per_second ": 65.508, "
eval_steps_per_second ": 4.099, "
epoch ": 3.0}"
```
