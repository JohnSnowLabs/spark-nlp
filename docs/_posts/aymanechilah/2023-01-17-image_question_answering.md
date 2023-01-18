---
layout: model
title: Document Visual Question Answering
author: John Snow Labs
name: image_question_answering
date: 2023-01-17
tags: [en, licensed, visual_question_answering, ocr]
task: Document Visual Question Answering
language: en
edition: Visual NLP 4.3.0
spark_version: 3.2.1
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Document understanding transformer (Donut) model pretrained for Document Visual Question Answering (DocVQA) task in the dataset is from Document Visual Question Answering [competition](https://rrc.cvc.uab.es/?ch=17) and consists of 50K questions defined on more than 12K documents. 
Donut is a new method of document understanding that utilizes an OCR-free end-to-end Transformer model. Donut does not require off-the-shelf OCR engines/APIs, yet it shows state-of-the-art performances on various visual document understanding tasks, such as visual document classification or information extraction (a.k.a. document parsing). Paper link [OCR-free Document Understanding Transformer](https://arxiv.org/abs/2111.15664) developed by Geewook Kim, Teakgyu Hong, Moonbin Yim, Jeongyeon Nam, Jinyoung Park, Jinyeong Yim, Wonseok Hwang, Sangdoo Yun, Dongyoon Han and Seunghyun Park.
DocVQA seeks to inspire a “purpose-driven” point of view in Document Analysis and Recognition research, where the document content is extracted and used to respond to high-level tasks defined by the human consumers of this information.

## Predicted Entities

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-ocr-workshop/blob/master/tutorials/Cards/SparkOcrVisualQuestionAnswering.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/ocr/docvqa_donut_base_en_4.3.0_3.0_1673269990044.zip){:.button.button-orange.button-orange-trans.arr.button-icon}


## How to use

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python

    from pyspark.ml import PipelineModel
    from sparkocr.transformers import *

    imagePath = "path to image"
    bin_df = spark.read.format("binaryFile").load(imagePath)
    image_df = BinaryToImage().transform(bin_df)

    questions = [["question 1", "question 2", "question X"]]
    questions_df = spark.createDataFrame([questions])
    questions_df = questions_df.withColumnRenamed("_1", "questions")
    image_and_questions = bin_df.join(questions_df)

    binary_to_image = BinaryToImage()\
        .setOutputCol("image") \
        .setImageType(ImageType.TYPE_3BYTE_BGR)

    visual_question_answering = VisualQuestionAnswering()\
        .pretrained("docvqa_donut_base", "en", "clinical/ocr")\
        .setInputCol(["image"])\
        .setOutputCol("answers")\
        .setQuestionsCol("questions")
    
    # OCR pipeline
    pipeline = PipelineModel(stages=[
        binary_to_image,
        visual_question_answering
    ])

    results = pipeline.transform(image_and_questions).cache()
    results.select(results.answers).show(truncate=False)
```
```scala
import com.johnsnowlabs.ocr.transformers.*
import com.johnsnowlabs.ocr.OcrContext.implicits._

val imagePath = "path to image"
val imageDf = spark.read.format("binaryFile").load(imagePath)
val questions = Seq(Array("question 1", "question 2", "question X")).toDF("questions")
val imageWithQuestions = imageDf.join(questions)

val binaryToImage = new BinaryToImage().
  setOutputCol("image")

val visualQA = VisualQuestionAnswering.
  pretrained("docvqa_donut_base", "en", "clinical/ocr").
  setQuestionsCol("questions")

val pipeline = new Pipeline()
  .setStages(Array(binaryToImage, visualQA))
  .fit(imageWithQuestions)

val tmp = pipeline.transform(imageWithQuestions).select("answers").collect()

// TODO broken!!
tmp.head.get(0) match {
  case array:mutable.WrappedArray.ofRef[mutable.WrappedArray.ofRef[_]] =>
    assert(array.head.head.toString contains "11:44")
    assert(array.tail.head.head.toString contains "lee a.")
    assert(array.last.head.toString contains "trrf")
    }
    
```
</div>

## Example

### Input:
```bash
+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|questions                                                                                                                                                                                                                    |
+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|[ When it finish the Coffee Break?,  Who is giving the Introductory Remarks?, Who is going to take part of the individual interviews?
+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
```
![Screenshot](../../_examples_ocr/image12.png)


### Output:
```bash
+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|answers                                                                                                                                                                                                                    |
+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|[ When it finish the Coffee Break? ->  11:44 to 11:39 a.m.,  Who is giving the Introductory Remarks? ->  lee a. waller, trrf vice presi- dent,  Who is going to take part of the individual interviews? ->  trrf treasurer]|
+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
```



