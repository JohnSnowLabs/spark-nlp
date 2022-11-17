---
layout: model
title: DistilBERT Sequence Classification - Policy (distilbert_sequence_classifier_policy)
author: John Snow Labs
name: distilbert_sequence_classifier_policy
date: 2021-11-21
tags: [sequence_classification, en, english, open_source, distilbert, policy, political_categories]
task: Text Classification
language: en
edition: Spark NLP 3.3.3
spark_version: 3.0
supported: true
annotator: DistilBertForSequenceClassification
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model was trained on 129.669 manually annotated sentences to classify text into one of seven political categories: 'Economy', 'External Relations', 'Fabric of Society', 'Freedom and Democracy', 'Political System', 'Welfare and Quality of Life' or 'Social Groups'.

### Training data

Policy-DistilBERT-7d was trained on the English-speaking subset of the [Manifesto Project Dataset (MPDS2020a)](https://manifesto-project.wzb.eu/datasets). The model was trained on 129.669 sentences from 164 political manifestos from 55 political parties in 8 English-speaking countries (Australia, Canada, Ireland, Israel, New Zealand, South Africa, United Kingdom, United States). The manifestos were published between 1992 - 2019. 

The Manifesto Project manually annotates individual sentences from political party manifestos in 7 main political domains: 'Economy', 'External Relations', 'Fabric of Society', 'Freedom and Democracy', 'Political System', 'Welfare and Quality of Life' or 'Social Groups' - see the [codebook](https://manifesto-project.wzb.eu/down/data/2020b/codebooks/codebook_MPDataset_MPDS2020b.pdf) for the exact definitions of each domain. 

### Limitations and bias

The model was trained on sentences in political manifestos from parties in the 8 countries mentioned above between 1992-2019, manually annotated by the [Manifesto Project](https://manifesto-project.wzb.eu/information/documents/information). The model output, therefore, reproduces the limitations of the dataset in terms of country coverage, time span, domain definitions, and potential biases of the annotators - as any supervised machine learning model would. Applying the model to other types of data (other types of texts, countries, etc.) will reduce performance.

## Predicted Entities

`Economy`, `External Relations`, `Fabric of Society`, `Freedom and Democracy`, `Political System`, `Welfare and Quality of Life`, `Social Groups`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/distilbert_sequence_classifier_policy_en_3.3.3_3.0_1637495279131.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

sequenceClassifier = DistilBertForSequenceClassification \
.pretrained('distilbert_sequence_classifier_policy', 'en') \
.setInputCols(['token', 'document']) \
.setOutputCol('class') \
.setMaxSentenceLength(512)

pipeline = Pipeline(stages=[
document_assembler, 
tokenizer,
sequenceClassifier    
])

example = spark.createDataFrame([['70-85% of the population needs to get vaccinated against the novel coronavirus to achieve herd immunity.']]).toDF("text")
result = pipeline.fit(example).transform(example)
```
```scala
val document_assembler = DocumentAssembler() 
.setInputCol("text") 
.setOutputCol("document")

val tokenizer = Tokenizer() 
.setInputCols("document") 
.setOutputCol("token")

val tokenClassifier = DistilBertForSequenceClassification.pretrained("distilbert_sequence_classifier_policy", "en")
.setInputCols("document", "token")
.setOutputCol("class")
.setMaxSentenceLength(512)

val pipeline = new Pipeline().setStages(Array(document_assembler, tokenizer, sequenceClassifier))

val example = Seq("70-85% of the population needs to get vaccinated against the novel coronavirus to achieve herd immunity.").toDS.toDF("text")

val result = pipeline.fit(example).transform(example)
```


{:.nlu-block}
```python
import nlu
nlu.load("en.classify.distilbert_sequence.policy").predict("""70-85% of the population needs to get vaccinated against the novel coronavirus to achieve herd immunity.""")
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|distilbert_sequence_classifier_policy|
|Compatibility:|Spark NLP 3.3.3+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[token, document]|
|Output Labels:|[class]|
|Language:|en|
|Case sensitive:|true|
|Max sentense length:|512|

## Data Source

[https://huggingface.co/MoritzLaurer/policy-distilbert-7d](https://huggingface.co/MoritzLaurer/policy-distilbert-7d)

## Benchmarking

```bash
The model was evaluated using 15% of the sentences (85-15 train-test split).

accuracy (balanced)   | F1 (weighted) | precision | recall | accuracy (not balanced) 
-------|---------|----------|---------|----------
0.745  | 0.773 | 0.772 | 0.771 | 0.771


Please note that the label distribution in the dataset is imbalanced:


Welfare and Quality of Life    0.327225
Economy                        0.259191
Fabric of Society              0.111800
Political System               0.095081
Social Groups                  0.094371
External Relations             0.063724
Freedom and Democracy          0.048608
```
