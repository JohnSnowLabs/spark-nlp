---
layout: model
title: Part of Speech for Swedish
author: John Snow Labs
name: pos_ud_tal
date: 2020-05-04 23:32:00 +0800
task: Part of Speech Tagging
language: sv
edition: Spark NLP 2.5.0
spark_version: 2.4
tags: [pos, es]
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
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/pos_ud_tal_sv_2.5.0_2.4_1588622711284.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/pos_ud_tal_sv_2.5.0_2.4_1588622711284.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

{:.h2_title}
## How to use 

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
...
pos = PerceptronModel.pretrained("pos_ud_tal", "sv") \
.setInputCols(["document", "token"]) \
.setOutputCol("pos")
nlp_pipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, pos])
light_pipeline = LightPipeline(nlp_pipeline.fit(spark.createDataFrame([['']]).toDF("text")))
results = light_pipeline.fullAnnotate("Förutom att vara kungen i norr är John Snow en engelsk läkare och en ledare inom utveckling av anestesi och medicinsk hygien.")
```

```scala
...
val pos = PerceptronModel.pretrained("pos_ud_tal", "sv")
.setInputCols(Array("document", "token"))
.setOutputCol("pos")
val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, tokenizer, pos))
val data = Seq("Förutom att vara kungen i norr är John Snow en engelsk läkare och en ledare inom utveckling av anestesi och medicinsk hygien.").toDF("text")
val result = pipeline.fit(data).transform(data)
```

{:.nlu-block}
```python
import nlu

text = ["""Förutom att vara kungen i norr är John Snow en engelsk läkare och en ledare inom utveckling av anestesi och medicinsk hygien."""]
pos_df = nlu.load('sv.pos.ud_tal').predict(text, output_level='token')
pos_df
```

</div>

{:.h2_title}
## Results

```bash
[Row(annotatorType='pos', begin=0, end=6, result='ADP', metadata={'word': 'Förutom'}),
Row(annotatorType='pos', begin=8, end=10, result='PART', metadata={'word': 'att'}),
Row(annotatorType='pos', begin=12, end=15, result='AUX', metadata={'word': 'vara'}),
Row(annotatorType='pos', begin=17, end=22, result='NOUN', metadata={'word': 'kungen'}),
Row(annotatorType='pos', begin=24, end=24, result='ADP', metadata={'word': 'i'}),
...]
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|pos_ud_tal|
|Type:|pos|
|Compatibility:|Spark NLP 2.5.0+|
|Edition:|Official|
|Input labels:|[token]|
|Output labels:|[pos]|
|Language:|sv|
|Case sensitive:|false|
|License:|Open Source|

{:.h2_title}
## Data Source
The model is imported from [https://universaldependencies.org](https://universaldependencies.org)