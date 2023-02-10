---
layout: model
title: Drug Spell Checker
author: John Snow Labs
name: spellcheck_drug_norvig
date: 2021-09-15
tags: [spell, spell_checker, clinical, en, licensed, drug]
task: Spell Check
language: en
edition: Healthcare NLP 3.2.2
spark_version: 3.0
supported: true
annotator: NorvigSweetingModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model detects and corrects spelling errors of drugs in your input text based on Norvig's approach.

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/spellcheck_drug_norvig_en_3.2.2_3.0_1631700986904.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/spellcheck_drug_norvig_en_3.2.2_3.0_1631700986904.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = DocumentAssembler()\
.setInputCol("text")\
.setOutputCol("document")

tokenizer = Tokenizer()\
.setInputCols("document")\
.setOutputCol("token")

spell = NorvigSweetingModel.pretrained("spellcheck_drug_norvig", "en", "clinical/models")\
.setInputCols("token")\
.setOutputCol("spell")\


pipeline = Pipeline(
stages = [
documentAssembler,    
tokenizer,
spell])

model = pipeline.fit(spark.createDataFrame([['']]).toDF('text'))
lp = LightPipeline(model)

result = lp.annotate("You have to take Neutrcare and colfosrinum and a bit of Fluorometholne & Ribotril")
```
```scala
val documentAssembler = new DocumentAssembler()\
.setInputCol("text")\
.setOutputCol("document")

val tokenizer = new Tokenizer()\
.setInputCols("document")\
.setOutputCol("token")

val spell = new NorvigSweetingModel.pretrained("spellcheck_drug_norvig", "en", "clinical/models")\
.setInputCols("token")\
.setOutputCol("spell")\

val pipeline = new Pipeline().setStages(Array(documentAssembler,tokenizer,spell))

val model = pipeline.fit(spark.createDataFrame([['']]).toDF('text'))
val lp = new LightPipeline(model)

val result = lp.annotate("You have to take Neutrcare and colfosrinum and a bit of Fluorometholne & Ribotril")
```


{:.nlu-block}
```python
import nlu
nlu.load("en.spell.drug_norvig").predict("""You have to take Neutrcare and colfosrinum and a bit of Fluorometholne & Ribotril""")
```

</div>

## Results

```bash
Original text  : You have to take Neutrcare and colfosrinum and a bit of fluorometholne & Ribotril

Corrected text : You have to take Neutracare and colforsinum and a bit of fluorometholone & Rivotril
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|spellcheck_drug_norvig|
|Compatibility:|Healthcare NLP 3.2.2+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[token]|
|Output Labels:|[spell]|
|Language:|en|
|Case sensitive:|true|
