---
layout: model
title: Sentence Detection in Sindhi Text
author: John Snow Labs
name: sentence_detector_dl
date: 2021-08-30
tags: [sd, open_source, sentence_detection]
task: Sentence Detection
language: sd
edition: Spark NLP 3.2.0
spark_version: 3.0
supported: true
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

SentenceDetectorDL (SDDL) is based on a general-purpose neural network model for sentence boundary detection. The task of sentence boundary detection is to identify sentences within a text. Many natural language processing tasks take a sentence as an input unit, such as part-of-speech tagging, dependency parsing, named entity recognition or machine translation.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Public/9.SentenceDetectorDL.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/sentence_detector_dl_sd_3.2.0_3.0_1630337452693.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/sentence_detector_dl_sd_3.2.0_3.0_1630337452693.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documenter = DocumentAssembler()\
.setInputCol("text")\
.setOutputCol("document")

sentencerDL = SentenceDetectorDLModel\
.pretrained("sentence_detector_dl", "sd") \
.setInputCols(["document"]) \
.setOutputCol("sentences")

sd_model = LightPipeline(PipelineModel(stages=[documenter, sentencerDL]))

sd_model.fullAnnotate("""readingولي رھيا آھن ھڪڙو وڏو ذريعو انگريزي پڙھڻ جا پيراگراف؟ توھان صحيح ھن place تي آيا آھيو. هڪ تازي تحقيق مطابق ا today's جي نوجوانن ۾ پڙهڻ جي عادت تيزيءَ سان گهٽجي رهي آهي. اھي نٿا ڏئي سگھن انگريزي ڏنل پيراگراف تي ڪجھ سيڪنڊن کان و forيڪ لاءِ. پڻ ، پڙهڻ هو ۽ آهي هڪ لازمي حصو س allني مقابلي واري امتحانن جو. تنھنڪري ، توھان پنھنجي پڙھڻ جي صلاحيتن کي ڪيئن بھتر ڪريو ٿا؟ ھن سوال جو جواب اصل ۾ ھڪڙو questionيو سوال آھي: پڙھڻ جي صلاحيتن جو استعمال ا آھي؟ پڙهڻ جو بنيادي مقصد آهي ’احساس ڪرڻ‘.""")

```
```scala
val documenter = DocumentAssembler()
.setInputCol("text")
.setOutputCol("document")

val model = SentenceDetectorDLModel.pretrained("sentence_detector_dl", "sd")
	.setInputCols(Array("document"))
	.setOutputCol("sentence")

val pipeline = new Pipeline().setStages(Array(documenter, model))

val data = Seq("readingولي رھيا آھن ھڪڙو وڏو ذريعو انگريزي پڙھڻ جا پيراگراف؟ توھان صحيح ھن place تي آيا آھيو. هڪ تازي تحقيق مطابق ا today's جي نوجوانن ۾ پڙهڻ جي عادت تيزيءَ سان گهٽجي رهي آهي. اھي نٿا ڏئي سگھن انگريزي ڏنل پيراگراف تي ڪجھ سيڪنڊن کان و forيڪ لاءِ. پڻ ، پڙهڻ هو ۽ آهي هڪ لازمي حصو س allني مقابلي واري امتحانن جو. تنھنڪري ، توھان پنھنجي پڙھڻ جي صلاحيتن کي ڪيئن بھتر ڪريو ٿا؟ ھن سوال جو جواب اصل ۾ ھڪڙو questionيو سوال آھي: پڙھڻ جي صلاحيتن جو استعمال ا آھي؟ پڙهڻ جو بنيادي مقصد آهي ’احساس ڪرڻ‘.").toDF("text")

val result = pipeline.fit(data).transform(data)

```

{:.nlu-block}
```python
import nlu

nlu.load('sd.sentence_detector').predict("readingولي رھيا آھن ھڪڙو وڏو ذريعو انگريزي پڙھڻ جا پيراگراف؟ توھان صحيح ھن place تي آيا آھيو. هڪ تازي تحقيق مطابق ا today's جي نوجوانن ۾ پڙهڻ جي عادت تيزيءَ سان گهٽجي رهي آهي. اھي نٿا ڏئي سگھن انگريزي ڏنل پيراگراف تي ڪجھ سيڪنڊن کان و forيڪ لاءِ. پڻ ، پڙهڻ هو ۽ آهي هڪ لازمي حصو س allني مقابلي واري امتحانن جو. تنھنڪري ، توھان پنھنجي پڙھڻ جي صلاحيتن کي ڪيئن بھتر ڪريو ٿا؟ ھن سوال جو جواب اصل ۾ ھڪڙو questionيو سوال آھي: پڙھڻ جي صلاحيتن جو استعمال ا آھي؟ پڙهڻ جو بنيادي مقصد آهي ’احساس ڪرڻ‘.", output_level ='sentence')  

```
</div>

## Results

```bash
+--------------------------------------------------------------------------------------------------------------+
|result                                                                                                        |
+--------------------------------------------------------------------------------------------------------------+
|[readingولي رھيا آھن ھڪڙو وڏو ذريعو انگريزي پڙھڻ جا پيراگراف؟ توھان صحيح ھن place تي آيا آھيو.]               |
|[هڪ تازي تحقيق مطابق ا today's جي نوجوانن ۾ پڙهڻ جي عادت تيزيءَ سان گهٽجي رهي آهي.]                           |
|[اھي نٿا ڏئي سگھن انگريزي ڏنل پيراگراف تي ڪجھ سيڪنڊن کان و forيڪ لاءِ.]                                       |
|[پڻ ، پڙهڻ هو ۽ آهي هڪ لازمي حصو س allني مقابلي واري امتحانن جو.]                                             |
|[تنھنڪري ، توھان پنھنجي پڙھڻ جي صلاحيتن کي ڪيئن بھتر ڪريو ٿا؟ ھن سوال جو جواب اصل ۾ ھڪڙو questionيو سوال آھي:]|
|[پڙھڻ جي صلاحيتن جو استعمال ا آھي؟ پڙهڻ جو بنيادي مقصد آهي ’احساس ڪرڻ‘.]                                      |
+--------------------------------------------------------------------------------------------------------------+


```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sentence_detector_dl|
|Compatibility:|Spark NLP 3.2.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document]|
|Output Labels:|[sentences]|
|Language:|sd|

## Benchmarking

```bash
label  Accuracy  Recall   Prec   F1  
0      0.98      1.00     0.96   0.98
```