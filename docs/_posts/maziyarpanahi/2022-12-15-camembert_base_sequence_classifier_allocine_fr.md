---
layout: model
title: CamemBERT Sequence Classification Base - French Allociné (camembert_base_sequence_classifier_allocine)
author: John Snow Labs
name: camembert_base_sequence_classifier_allocine
date: 2022-12-15
tags: [open_source, camembert, sequence_classification, fr, french, allocine, sentiment, sentiment_analysis, tensorflow]
task: Sentiment Analysis
language: fr
edition: Spark NLP 4.2.5
spark_version: 3.0
supported: true
engine: tensorflow
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

CamemBertForSequenceClassification can load CamemBERT model with sequence classification/regression head on top (a linear layer on top of the pooled output) e.g. for multi-class document classification tasks.

camembert_base_sequence_classifier_allocine is a fine-tuned CamemBERT model on French Allociné dataset that is ready to be used for Sequence Classification sentiment analysis task abd it achieves state-of-the-art performance.

We used TFCamembertForSequenceClassification in Transformers to train this model and used CamemBertForSequenceClassification annotator in Spark NLP 🚀 for prediction at scale!

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/camembert_base_sequence_classifier_allocine_fr_4.2.5_3.0_1671101501132.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/camembert_base_sequence_classifier_allocine_fr_4.2.5_3.0_1671101501132.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

sequenceClassifier = CamemBertForSequenceClassification.pretrained("camembert_base_sequence_classifier_allocine", "fr")\ 
    .setInputCols(["document", "token"])\ 
    .setOutputCol("class")\ 
    .setCaseSensitive(True)\ 
    .setMaxSentenceLength(512) 

pipeline = Pipeline(stages=[
    document_assembler,
    tokenizer,
    sequenceClassifier
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

val sequenceClassifier = CamemBertForSequenceClassification.pretrained("camembert_base_sequence_classifier_allocine", "fr")
    .setInputCols("document", "token")
    .setOutputCol("class")
    .setCaseSensitive(true)
    .setMaxSentenceLength(512)

val pipeline = new Pipeline().setStages(Array(document_assembler, tokenizer, sequenceClassifier))

val example = Seq("I really liked that movie!").toDS.toDF("text")

val result = pipeline.fit(example).transform(example)
```


{:.nlu-block}
```python
import nlu
nlu.load("fr.classify.camembert.base").predict("""I really liked that movie!""")
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|camembert_base_sequence_classifier_allocine|
|Compatibility:|Spark NLP 4.2.5+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[token, document]|
|Output Labels:|[class]|
|Language:|fr|
|Size:|415.3 MB|
|Case sensitive:|true|
|Max sentence length:|512|

## References

[https://huggingface.co/datasets/allocine](https://huggingface.co/datasets/allocine)