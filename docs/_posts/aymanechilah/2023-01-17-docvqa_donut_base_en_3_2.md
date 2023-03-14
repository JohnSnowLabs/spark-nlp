---
layout: model
title: Document Visual Question Answering with DONUT
author: John Snow Labs
name: docvqa_donut_base
date: 2023-01-17
tags: [en, licensed]
task: Document Visual Question Answering
language: en
nav_key: models
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
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-ocr-workshop/blob/master/jupyter/Cards/SparkOcrVisualQuestionAnswering.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/ocr/docvqa_donut_base_en_4.3.0_3.0_1673269990044.zip){:.button.button-orange.button-orange-trans.arr.button-icon}


## How to use

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
binary_to_image = BinaryToImage()\
    .setInputCol("content") \
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

test_image_path = pkg_resources.resource_filename('sparkocr', 'resources/ocr/vqa/agenda.png')
bin_df = spark.read.format("binaryFile").load(test_image_path)

questions = [["When it finish the Coffee Break?", "Who is giving the Introductory Remarks?", "Who is going to take part of the individual interviews?"]]
questions_df = spark.createDataFrame([questions])
questions_df = questions_df.withColumnRenamed("_1", "questions")
image_and_questions = bin_df.join(questions_df)

results = pipeline.transform(image_and_questions).cache()
results.select(results.answers).show(truncate=False)
```
```scala
val binary_to_image = new BinaryToImage()
    .setInputCol("content") 
    .setOutputCol("image") 
    .setImageType(ImageType.TYPE_3BYTE_BGR)

val visual_question_answering = VisualQuestionAnswering()
    .pretrained("docvqa_donut_base", "en", "clinical/ocr")
    .setInputCol(Array("image"))
    .setOutputCol("answers")
    .setQuestionsCol("questions")

# OCR pipeline
val pipeline = new PipelineModel().setStages(Array(
    binary_to_image, 
    visual_question_answering))

val test_image_path = pkg_resources.resource_filename("sparkocr", "resources/ocr/vqa/agenda.png")
val bin_df = spark.read.format("binaryFile").load(test_image_path)

val questions = Array("When it finish the Coffee Break?", "Who is giving the Introductory Remarks?", "Who is going to take part of the individual interviews?")
val questions_df = spark.createDataFrame(Array(questions))
val questions_df = questions_df.withColumnRenamed("_1", "questions")
val image_and_questions = bin_df.join(questions_df)

val results = pipeline.transform(image_and_questions).cache()
results.select(results.answers).show(truncate=False)
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
![Screenshot](/assets/images/examples_ocr/image12.png)


### Output:
```bash
+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|answers                                                                                                                                                                                                                    |
+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|[ When it finish the Coffee Break? ->  11:44 to 11:39 a.m.,  Who is giving the Introductory Remarks? ->  lee a. waller, trrf vice presi- dent,  Who is going to take part of the individual interviews? ->  trrf treasurer]|
+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
```



