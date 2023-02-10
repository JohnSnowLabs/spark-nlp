---
layout: model
title: Toxic Comment Classification - Small
author: John Snow Labs
name: multiclassifierdl_use_toxic_sm
date: 2021-01-21
task: Text Classification
language: en
edition: Spark NLP 2.7.1
spark_version: 2.4
tags: [open_source, en, text_classification]
supported: true
annotator: MultiClassifierDLModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Discussing things you care about can be difficult. The threat of abuse and harassment online means that many people stop expressing themselves and give up on seeking different opinions. Platforms struggle to effectively facilitate conversations, leading many communities to limit or completely shut down user comments.

The Conversation AI team, a research initiative founded by Jigsaw and Google (both a part of Alphabet) is working on tools to help improve the online conversation. One area of focus is the study of negative online behaviors, like toxic comments (i.e. comments that are rude, disrespectful, or otherwise likely to make someone leave a discussion). So far they’ve built a range of publicly available models served through the Perspective API, including toxicity. But the current models still make errors, and they don’t allow users to select which types of toxicity they’re interested in finding (e.g. some platforms may be fine with profanity, but not with other types of toxic content).

Automatically detect identity hate, insult, obscene, severe toxic, threat, or toxic content in SM comments using our out-of-the-box Spark NLP Multiclassifier DL.
We removed the records without any labels in this model. (only 14K+ comments were used to train this model)

## Predicted Entities

`toxic`, `severe_toxic`, `identity_hate`, `insult`, `obscene`, `threat`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/public/CLASSIFICATION_MULTILABEL_TOXIC/){:.button.button-orange}
[Open in Colab](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/CLASSIFICATION_MULTILABEL_TOXIC.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/multiclassifierdl_use_toxic_sm_en_2.7.1_2.4_1611230645484.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/multiclassifierdl_use_toxic_sm_en_2.7.1_2.4_1611230645484.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
document = DocumentAssembler()\
.setInputCol("text")\
.setOutputCol("document")

use = UniversalSentenceEncoder.pretrained() \
.setInputCols(["document"])\
.setOutputCol("use_embeddings")

docClassifier = MultiClassifierDLModel.pretrained("multiclassifierdl_use_toxic_sm") \
.setInputCols(["use_embeddings"])\
.setOutputCol("category")\
.setThreshold(0.5)

pipeline = Pipeline(
stages = [
document,
use,
docClassifier
])
```
```scala
val documentAssembler = new DocumentAssembler()
.setInputCol("text")
.setOutputCol("document")
.setCleanupMode("shrink")

val use = UniversalSentenceEncoder.pretrained()
.setInputCols("document")
.setOutputCol("use_embeddings")

val docClassifier = MultiClassifierDLModel.pretrained("multiclassifierdl_use_toxic_sm")
.setInputCols("use_embeddings")
.setOutputCol("category")
.setThreshold(0.5f)

val pipeline = new Pipeline()
.setStages(
Array(
documentAssembler,
use,
docClassifier
)
)
```


{:.nlu-block}
```python
import nlu
nlu.load("en.classify.toxic.sm").predict("""Put your text here.""")
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|multiclassifierdl_use_toxic_sm|
|Compatibility:|Spark NLP 2.7.1+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[use_embeddings]|
|Output Labels:|[category]|
|Language:|en|

## Data Source

https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/overview

## Benchmarking

```bash
Classification report: 
precision    recall  f1-score   support

0       0.56      0.30      0.39       127
1       0.71      0.70      0.70       761
2       0.76      0.72      0.74       824
3       0.55      0.21      0.31       147
4       0.79      0.38      0.51        50
5       0.94      1.00      0.97      1504

micro avg       0.83      0.80      0.81      3413
macro avg       0.72      0.55      0.60      3413
weighted avg       0.81      0.80      0.80      3413
samples avg       0.84      0.83      0.80      3413
```