---
layout: model
title: Part of Speech for Finnish
author: John Snow Labs
name: pos_ud_tdt
date: 2020-05-04 23:32:00 +0800
task: Part of Speech Tagging
language: fi
edition: Spark NLP 2.5.0
spark_version: 2.4
tags: [pos, fi]
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
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/pos_ud_tdt_fi_2.5.0_2.4_1588622348985.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/pos_ud_tdt_fi_2.5.0_2.4_1588622348985.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

{:.h2_title}
## How to use 

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
...
pos = PerceptronModel.pretrained("pos_ud_tdt", "fi") \
.setInputCols(["document", "token"]) \
.setOutputCol("pos")
nlp_pipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, pos])
light_pipeline = LightPipeline(nlp_pipeline.fit(spark.createDataFrame([['']]).toDF("text")))
results = light_pipeline.fullAnnotate("Sen lisäksi, että hän on pohjoisen kuningas, John Snow on englantilainen lääkäri ja johtava anestesian ja lääketieteellisen hygienian kehittämisessä.")
```

```scala
...
val pos = PerceptronModel.pretrained("pos_ud_tdt", "fi")
.setInputCols(Array("document", "token"))
.setOutputCol("pos")
val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, tokenizer, pos))
val data = Seq("Sen lisäksi, että hän on pohjoisen kuningas, John Snow on englantilainen lääkäri ja johtava anestesian ja lääketieteellisen hygienian kehittämisessä.").toDF("text")
val result = pipeline.fit(data).transform(data)
```

{:.nlu-block}
```python
import nlu

text = ["""Sen lisäksi, että hän on pohjoisen kuningas, John Snow on englantilainen lääkäri ja johtava anestesian ja lääketieteellisen hygienian kehittämisessä."""]
pos_df = nlu.load('fi.pos.ud_tdt').predict(text, output_level='token')
pos_df
```

</div>

{:.h2_title}
## Results

```bash
[Row(annotatorType='pos', begin=0, end=2, result='PRON', metadata={'word': 'Sen'}),
Row(annotatorType='pos', begin=4, end=10, result='ADP', metadata={'word': 'lisäksi'}),
Row(annotatorType='pos', begin=11, end=11, result='PUNCT', metadata={'word': ','}),
Row(annotatorType='pos', begin=13, end=16, result='SCONJ', metadata={'word': 'että'}),
Row(annotatorType='pos', begin=18, end=20, result='PRON', metadata={'word': 'hän'}),
...]
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|pos_ud_tdt|
|Type:|pos|
|Compatibility:|Spark NLP 2.5.0+|
|Edition:|Official|
|Input labels:|[token]|
|Output labels:|[pos]|
|Language:|fi|
|Case sensitive:|false|
|License:|Open Source|

{:.h2_title}
## Data Source
The model is imported from [https://universaldependencies.org](https://universaldependencies.org)