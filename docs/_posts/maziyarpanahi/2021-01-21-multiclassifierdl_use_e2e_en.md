---
layout: model
title: End-to-End (E2E) and data-driven NLG Challenge
author: John Snow Labs
name: multiclassifierdl_use_e2e
date: 2021-01-21
task: Text Classification
language: en
edition: Spark NLP 2.7.1
spark_version: 2.4
tags: [en, open_source, text_classification]
supported: true
annotator: MultiClassifierDLModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Natural language generation plays a critical role for Conversational Agents as it has a significant impact on a user’s impression of the system. This shared task focuses on recent end-to-end (E2E), data-driven NLG methods, which jointly learn sentence planning and surface realization from non-aligned data, e.g. (Wen et al., 2015; Mei et al., 2016; Dusek and Jurcicek, 2016; Lampouras and Vlachos, 2016), etc.

So far, E2E NLG approaches were limited to small, de-lexicalized data sets, e.g. BAGEL, SF Hotels/ Restaurants, or RoboCup. In this shared challenge, we will provide a new crowd-sourced data set of 50k instances in the restaurant domain, as described in (Novikova, Lemon, and Rieser, 2016). Each instance consists of a dialogue act-based meaning representation (MR) and up to 5 references in natural language. In contrast to previously used data, our data set includes additional challenges, such as open vocabulary, complex syntactic structures, and diverse discourse phenomena.

## Predicted Entities

`name[Bibimbap House]`,`name[Wildwood]`,`name[Clowns]`,`name[Cotto]`,`near[Burger King]`,`name[The Dumpling Tree]`,`name[The Vaults]`,`name[The Golden Palace]`,`near[Crowne Plaza Hotel]`,`name[The Rice Boat]`,`customer rating[high]`,`near[Avalon]`,`name[Alimentum]`,`near[The Bakers]`,`name[The Waterman]`,`near[Ranch]`,`name[The Olive Grove]`,`name[The Wrestlers]`,`name[The Eagle]`,`eatType[restaurant]`,`near[All Bar One]`,`customer rating[low]`,`near[Café Sicilia]`,`near[Yippee Noodle Bar]`,`food[Indian]`,`eatType[pub]`,`name[Green Man]`,`name[Strada]`,`near[Café Adriatic]`,`name[Loch Fyne]`,`eatType[coffee shop]`,`customer rating[5 out of 5]`,`near[Express by Holiday Inn]`,`food[French]`,`name[The Mill]`,`food[Japanese]`,`name[Travellers Rest Beefeater]`,`name[The Plough]`,`name[Cocum]`,`near[The Six Bells]`,`name[The Phoenix]`,`priceRange[cheap]`,`name[Midsummer House]`,`near[Rainbow Vegetarian Café]`,`near[The Rice Boat]`,`customer rating[3 out of 5]`,`customer rating[1 out of 5]`,`name[The Cricketers]`,`area[riverside]`,`priceRange[£20-25]`,`name[Blue Spice]`,`priceRange[moderate]`,`priceRange[less than £20]`,`priceRange[high]`,`name[Giraffe]`,`name[The Golden Curry]`,`customer rating[average]`,`name[The Twenty Two]`,`name[Aromi]`,`food[Fast food]`,`name[Browns Cambridge]`,`near[Café Rouge]`,`area[city centre]`,`familyFriendly[no]`,`food[Chinese]`,`name[Taste of Cambridge]`,`food[Italian]`,`name[Zizzi]`,`near[Raja Indian Cuisine]`,`priceRange[more than £30]`,`name[The Punter]`,`food[English]`,`near[Clare Hall]`,`near[The Portland Arms]`,`name[The Cambridge Blue]`,`near[The Sorrento]`,`near[Café Brazil]`,`familyFriendly[yes]`,`name[Fitzbillies]`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/multiclassifierdl_use_e2e_en_2.7.1_2.4_1611233305602.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/multiclassifierdl_use_e2e_en_2.7.1_2.4_1611233305602.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

docClassifier = MultiClassifierDLModel.pretrained("multiclassifierdl_use_e2e") \
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

val docClassifier = MultiClassifierDLModel.pretrained("multiclassifierdl_use_e2e")
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
nlu.load("en.classify.e2e").predict("""Put your text here.""")
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|multiclassifierdl_use_e2e|
|Compatibility:|Spark NLP 2.7.1+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[use_embeddings]|
|Output Labels:|[category]|
|Language:|en|

## Data Source

http://www.macs.hw.ac.uk/InteractionLab/E2E/

## Benchmarking

```bash
Summary Statistics
Accuracy = 0.6366936009433872
F1 measure = 0.7561380632067716
Precision = 0.8678456763698633
Recall = 0.6911700403620353
Micro F1 measure = 0.7750978356361313
Micro precision = 0.8694288913773797
Micro recall = 0.6992326812925538
```