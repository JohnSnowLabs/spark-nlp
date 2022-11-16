---
layout: model
title: Korean Word Segmentation
author: John Snow Labs
name: wordseg_kaist_ud
date: 2021-01-03
task: Word Segmentation
language: ko
edition: Spark NLP 2.7.0
spark_version: 2.4
tags: [open_source, word_segmentation, ko]
supported: true
annotator: WordSegmenterModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

WordSegmenterModel (WSM) is based on maximum entropy probability model to detect word boundaries in Chinese text. Chinese text is written without white space between the words, and a computer-based application cannot know _a priori_ which sequence of ideograms form a word. In many natural language processing tasks such as part-of-speech (POS) and named entity recognition (NER) require word segmentation as a initial step.


References:

- Xue, Nianwen. "Chinese word segmentation as character tagging." International Journal of Computational Linguistics & Chinese Language Processing, Volume 8, Number 1, February 2003: Special Issue on Word Formation and Chinese Language Processing. 2003.).

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/wordseg_kaist_ud_ko_2.7.0_2.4_1609693294761.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use

Use as part of an nlp pipeline as a substitute of the Tokenizer stage.

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
document_assembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")
    
word_segmenter = WordSegmenterModel.pretrained('wordseg_kaist_ud', 'ko')\
.setInputCols("document")\
.setOutputCol("token")
pipeline = Pipeline(stages=[
document_assembler,
word_segmenter
])
model = pipeline.fit(spark.createDataFrame([[""]]).toDF("text"))
example = spark.createDataFrame([['비파를탄주하는그늙은명인의시는아름다운화음이었고완벽한음악으로순간적인조화를이룬세계의울림이었다.']], ["text"])
result = model.transform(example)
```

```scala
val document_assembler = DocumentAssembler()
        .setInputCol("text")
        .setOutputCol("document")
        
val word_segmenter = WordSegmenterModel.pretrained("wordseg_kaist_ud", "ko")
.setInputCols("document")
.setOutputCol("token")
val pipeline = new Pipeline().setStages(Array(document_assembler, word_segmenter))
val data = Seq("비파를탄주하는그늙은명인의시는아름다운화음이었고완벽한음악으로순간적인조화를이룬세계의울림이었다.").toDF("text")
val result = pipeline.fit(data).transform(data)
```

{:.nlu-block}
```python
import nlu

text = ["""비파를탄주하는그늙은명인의시는아름다운화음이었고완벽한음악으로순간적인조화를이룬세계의울림이었다."""]
token_df = nlu.load('ko.segment_words').predict(text, output_level='token')
token_df
```

</div>

## Results

```bash
+-------------------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------------------------------------+
|text                                                                                             |result                                                                                                                           |
+-------------------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------------------------------------+
|비파를탄주하는그늙은명인의시는아름다운화음이었고완벽한음악으로순간적인조화를이룬세계의울림이었다.|[비파를, 탄주하는, 그, 늙은, 명인의, 시는, 아름다운, 화음이었고, 완벽한, 음악으로, 순간적인, 조화를, 이룬, 세계의, 울림이었다, .]|
+-------------------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------------------------------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|wordseg_kaist_ud|
|Compatibility:|Spark NLP 2.7.0+|
|Edition:|Official|
|Input Labels:|[document]|
|Output Labels:|[token]|
|Language:|ko|

## Data Source

We trained the model using the [Universal Dependenicies](universaldependencies.org) data set from Korea Advanced Institute of Science and Technology (KAIST-UD).

Reference:

> Building Universal Dependency Treebanks in Korean, Jayeol Chun, Na-Rae Han, Jena D. Hwang, and Jinho D. Choi. 
In Proceedings of the 11th International Conference on Language Resources and Evaluation, LREC'18, Miyazaki, Japan, 2018.

## Benchmarking

```bash
| Model         | precision    | recall       | f1-score     |
|---------------|--------------|--------------|--------------|
| KO_KAIST-UD   |      0,6966  |      0,7779  |      0,7350  |
```
