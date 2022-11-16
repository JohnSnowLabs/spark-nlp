---
layout: model
title: Question classification of open-domain and fact-based questions Pipeline - TREC50
author: John Snow Labs
name: classifierdl_use_trec50_pipeline
date: 2021-01-08
task: [Text Classification, Pipeline Public]
language: en
edition: Spark NLP 2.7.1
spark_version: 2.4
tags: [classifier, text_classification, pipeline, en, open_source]
supported: true
annotator: PipelineModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Classify open-domain, fact-based questions into one of the following broad semantic categories: Abbreviation, Description, Entities, Human Beings, Locations or Numeric Values.

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/public/CLASSIFICATION_EN_TREC/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/CLASSIFICATION_EN_TREC.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/classifierdl_use_trec50_pipeline_en_2.7.1_2.4_1610119592993.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline 
pipeline = PretrainedPipeline("classifierdl_use_trec50_pipeline", lang = "en") 

```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline
val pipeline = new PretrainedPipeline("classifierdl_use_trec50_pipeline", lang = "en")

```


{:.nlu-block}
```python
import nlu
nlu.load("en.classify.trec50.component_list").predict("""Put your text here.""")
```

</div>

## Results

```bash
+------------------------------------------------------------------------------------------------+------------+
|document                                                                                        |class       |
+------------------------------------------------------------------------------------------------+------------+
|When did the construction of stone circles begin in the UK?                                     | NUM_date   |
+------------------------------------------------------------------------------------------------+------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|classifierdl_use_trec50_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 2.7.1+|
|Edition:|Official|
|Language:|en|