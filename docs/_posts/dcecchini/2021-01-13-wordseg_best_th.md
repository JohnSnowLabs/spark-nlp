---
layout: model
title: Thai Word Segmentation
author: John Snow Labs
name: wordseg_best
date: 2021-01-13
task: Word Segmentation
language: th
edition: Spark NLP 2.7.0
spark_version: 2.4
tags: [th, word_segmentation, open_source]
supported: true
annotator: WordSegmenterModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

WordSegmenterModel (WSM) is based on maximum entropy probability model to detect word boundaries in Thai text. Thai text is written without white space between the words, and a computer-based application cannot know _a priori_ which sequence of ideograms form a word. In many natural language processing tasks such as part-of-speech (POS) and named entity recognition (NER) require word segmentation as a initial step.


References:

- Xue, Nianwen. "Chinese word segmentation as character tagging." International Journal of Computational Linguistics & Chinese Language Processing, Volume 8, Number 1, February 2003: Special Issue on Word Formation and Chinese Language Processing. 2003.).

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/wordseg_best_th_2.7.0_2.4_1610543628078.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use

Use as part of an nlp pipeline as a substitute of the Tokenizer stage.

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
document_assembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")
    
word_segmenter = WordSegmenterModel.pretrained('wordseg_best', 'th')\
        .setInputCols("document")\
        .setOutputCol("token")       
pipeline = Pipeline(stages=[document_assembler, word_segmenter])
example = spark.createDataFrame([['จวนจะถึงร้านที่คุณจองโต๊ะไว้แล้วจ้ะ']], ["text"])
result = pipeline.fit(example ).transform(example)
```

```scala
val document_assembler = DocumentAssembler()
        .setInputCol("text")
        .setOutputCol("document")
        
val document_assembler = DocumentAssembler()
        .setInputCol("text")
        .setOutputCol("document")
val word_segmenter = WordSegmenterModel.pretrained("wordseg_best", "th")
        .setInputCols("document")
        .setOutputCol("token")
val pipeline = new Pipeline().setStages(Array(document_assembler, word_segmenter))
val data = Seq("จวนจะถึงร้านที่คุณจองโต๊ะไว้แล้วจ้ะ").toDF("text")
val result = pipeline.fit(data).transform(data)
```

{:.nlu-block}
```python
import nlu

text = ["""Mona Lisa เป็นภาพวาดสีน้ำมันในศตวรรษที่ 16 ที่สร้างโดย Leonardo จัดขึ้นที่พิพิธภัณฑ์ลูฟร์ในปารีส"""]
token_df = nlu.load('th.segment_words').predict(text)
token_df
```

</div>

## Results

```bash
+-----------------------------------+---------------------------------------------------------+
|text                               |result                                                   |
+-----------------------------------+---------------------------------------------------------+
|จวนจะถึงร้านที่คุณจองโต๊ะไว้แล้วจ้ะ|[จวน, จะ, ถึง, ร้าน, ที่, คุณ, จอง, โต๊ะ, ไว้, แล้ว, จ้ะ]|
+-----------------------------------+---------------------------------------------------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|wordseg_best|
|Compatibility:|Spark NLP 2.7.0+|
|Edition:|Official|
|Input Labels:|[document]|
|Output Labels:|[token]|
|Language:|th|

## Data Source

The model was trained on the [BEST](http://thailang.nectec.or.th/best) corpus from the National Electronics and Computer Technology Center (NECTEC).

References:

> - Krit Kosawat, Monthika Boriboon, Patcharika Chootrakool, Ananlada Chotimongkol, Supon Klaithin, Sarawoot Kongyoung, Kanyanut Kriengket, Sitthaa Phaholphinyo, Sumonmas Purodakananda, Tipraporn Thanakulwarapas, and Chai Wutiwiwatchai, "BEST 2009: Thai word segmentation software contest," in Proc. 8th Int. Symp. Natural Language Process. (SNLP), Bangkok, Thailand, Oct.20-22, 2009, pp.83-88.
> - Monthika Boriboon, Kanyanut Kriengket, Patcharika Chootrakool, Sitthaa Phaholphinyo, Sumonmas Purodakananda, Tipraporn Thanakulwarapas, and Krit Kosawat, "BEST corpus development and analysis," in Proc. 2nd Int. Conf. Asian Language Process. (IALP), Singapore, Dec.7-9, 2009, pp.322-327.

## Benchmarking

```bash
| Model        | precision | recall | f1-score |
|--------------|-----------|--------|----------|
| WORDSEG_BEST | 0.4791    | 0.6245 | 0.5422   |
```