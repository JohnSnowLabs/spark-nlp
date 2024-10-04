---
layout: model
title: BLIP Question Answering
author: John Snow Labs
name: blip_vqa_base
date: 2024-10-03
tags: [en, open_source, tensorflow]
task: Question Answering
language: en
edition: Spark NLP 5.5.0
spark_version: 3.4
supported: true
engine: tensorflow
annotator: BLIPForQuestionAnswering
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

BLIP Model for visual question answering. The model consists of a vision encoder, a text encoder as well as a text decoder. The vision encoder will encode the input image, the text encoder will encode the input question together with the encoding of the image, and the text decoder will output the answer to the question.

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://github.com/JohnSnowLabs/spark-nlp/blob/master/examples/python/transformers/HuggingFace_in_Spark_NLP_BLIPForQuestionAnswering.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/blip_vqa_base_en_5.5.0_3.4_1727997969354.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/blip_vqa_base_en_5.5.0_3.4_1727997969354.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use

To proceed, please create a DataFrame with two columns:

- An image column that contains the file path for each image in the directory.
- A text column where you can input the specific question you would like to ask about each image.

For example:

```python
from pyspark.sql.functions import lit

images_path = "./images/"
image_df = spark.read.format("image").load(path=images_path)

test_df = image_df.withColumn("text", lit("What's this picture about?"))
test_df.show()
```

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
imageAssembler = ImageAssembler() \
  .setInputCol("image") \
  .setOutputCol("image_assembler") \

imageClassifier = BLIPForQuestionAnswering.load("./{}_spark_nlp".format(MODEL_NAME)) \
  .setInputCols("image_assembler") \
  .setOutputCol("answer") \
  .setSize(384)

pipeline = Pipeline(
    stages=[
        imageAssembler,
        imageClassifier,
    ]
)

model = pipeline.fit(test_df)
result = model.transform(test_df)
result.select("image_assembler.origin", "answer.result").show(truncate = False)
```
```scala
val imageAssembler: ImageAssembler = new ImageAssembler()
      .setInputCol("image")
      .setOutputCol("image_assembler")

val loadModel = BLIPForQuestionAnswering
  .pretrained()
  .setInputCols("image_assembler")
  .setOutputCol("answer")
  .setSize(384)

val newPipeline: Pipeline =
  new Pipeline().setStages(Array(imageAssembler, loadModel))

newPipeline.fit(testDF)
val result = model.transform(testDF)

result.select("image_assembler.origin", "answer.result").show(truncate = false)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|blip_vqa_base|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|1.4 GB|