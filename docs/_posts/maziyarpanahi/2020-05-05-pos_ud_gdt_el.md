---
layout: model
title: Part of Speech for Greek
author: John Snow Labs
name: pos_ud_gdt
date: 2020-05-05 16:56:00 +0800
task: Part of Speech Tagging
language: el
edition: Spark NLP 2.5.0
spark_version: 2.4
tags: [pos, el]
supported: true
annotator: PerceptronModel
article_header:
   type: cover
use_language_switcher: "Python-Scala-Java"
---

{:.h2_title}
## Description
This model annotates the part of speech of tokens in a text. The [parts of speech](https://universaldependencies.org/u/pos/) annotated include PRON (pronoun), CCONJ (coordinating conjunction), and 15 others. The part of speech model is useful for extracting the grammatical structure of a piece of text automatically.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://githubtocolab.com/JohnSnowLabs/spark-nlp-workshop/blob/2da56c087da53a2fac1d51774d49939e05418e57/tutorials/Certification_Trainings/Public/6.Playground_DataFrames.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/pos_ud_gdt_el_2.5.0_2.4_1588686949851.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/pos_ud_gdt_el_2.5.0_2.4_1588686949851.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

{:.h2_title}
## How to use 

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
...
pos = PerceptronModel.pretrained("pos_ud_gdt", "el") \
    .setInputCols(["document", "token"]) \
    .setOutputCol("pos")
nlp_pipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, pos])
light_pipeline = LightPipeline(nlp_pipeline.fit(spark.createDataFrame([['']]).toDF("text")))
results = light_pipeline.fullAnnotate("Εκτός από το ότι είναι ο βασιλιάς του Βορρά, ο John Snow είναι Άγγλος γιατρός και ηγέτης στην ανάπτυξη της αναισθησίας και της ιατρικής υγιεινής.")
```

```scala
...
val pos = PerceptronModel.pretrained("pos_ud_gdt", "el")
    .setInputCols(Array("document", "token"))
    .setOutputCol("pos")
val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, tokenizer, pos))
val data = Seq("Εκτός από το ότι είναι ο βασιλιάς του Βορρά, ο John Snow είναι Άγγλος γιατρός και ηγέτης στην ανάπτυξη της αναισθησίας και της ιατρικής υγιεινής.").toDF("text")
val result = pipeline.fit(data).transform(data)
```

{:.nlu-block}
```python
import nlu

text = ["""Εκτός από το ότι είναι ο βασιλιάς του Βορρά, ο John Snow είναι Άγγλος γιατρός και ηγέτης στην ανάπτυξη της αναισθησίας και της ιατρικής υγιεινής."""]
pos_df = nlu.load('el.pos.ud_gdt').predict(text)
pos_df
```

</div>

{:.h2_title}
## Results

```bash
[Row(annotatorType='pos', begin=0, end=4, result='ADV', metadata={'word': 'Εκτός'}),
Row(annotatorType='pos', begin=6, end=8, result='ADP', metadata={'word': 'από'}),
Row(annotatorType='pos', begin=10, end=11, result='DET', metadata={'word': 'το'}),
Row(annotatorType='pos', begin=13, end=15, result='SCONJ', metadata={'word': 'ότι'}),
Row(annotatorType='pos', begin=17, end=21, result='AUX', metadata={'word': 'είναι'}),
...]
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|pos_ud_gdt|
|Type:|pos|
|Compatibility:|Spark NLP 2.5.0+|
|Edition:|Official|
|Input labels:|[token]|
|Output labels:|[pos]|
|Language:|el|
|Case sensitive:|false|
|License:|Open Source|

{:.h2_title}
## Data Source
The model is imported from [https://universaldependencies.org](https://universaldependencies.org)