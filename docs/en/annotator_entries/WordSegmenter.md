{%- capture title -%}
WordSegmenter
{%- endcapture -%}

{%- capture model_description -%}
WordSegmenter which tokenizes non-english or non-whitespace separated texts.

Many languages are not whitespace separated and their sentences are a concatenation of many symbols, like Korean,
Japanese or Chinese. Without understanding the language, splitting the words into their corresponding tokens is
impossible. The WordSegmenter is trained to understand these languages and plit them into semantically correct parts.

This is the instantiated model of the WordSegmenterApproach.
For training your own model, please see the documentation of that class.

Pretrained models can be loaded with `pretrained` of the companion object:
```
val wordSegmenter = WordSegmenterModel.pretrained()
  .setInputCols("document")
  .setOutputCol("words_segmented")
```
The default model is `"wordseg_pku"`, default language is `"zh"`, if no values are provided.
For available pretrained models please see the [Models Hub](https://nlp.johnsnowlabs.com/models?task=Word+Segmentation).

For extended examples of usage, see the [Spark NLP Workshop](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/jupyter/annotation/chinese/word_segmentation/words_segmenter_demo.ipynb)
and the [WordSegmenterTest](https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/annotators/WordSegmenterTest.scala).
{%- endcapture -%}

{%- capture model_input_anno -%}
DOCUMENT
{%- endcapture -%}

{%- capture model_output_anno -%}
TOKEN
{%- endcapture -%}

{%- capture model_python_example -%}
import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline

documentAssembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")

wordSegmenter = WordSegmenterModel.pretrained() \
    .setInputCols(["document"]) \
    .setOutputCol("token")

pipeline = Pipeline().setStages([
    documentAssembler,
    wordSegmenter
])

data = spark.createDataFrame([["然而，這樣的處理也衍生了一些問題。"]]).toDF("text")
result = pipeline.fit(data).transform(data)

result.select("token.result").show(truncate=False)
+--------------------------------------------------------+
|result                                                  |
+--------------------------------------------------------+
|[然而, ，, 這樣, 的, 處理, 也, 衍生, 了, 一些, 問題, 。    ]|
+--------------------------------------------------------+

{%- endcapture -%}

{%- capture model_scala_example -%}
import spark.implicits._
import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.annotator.WordSegmenterModel
import org.apache.spark.ml.Pipeline

val documentAssembler = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val wordSegmenter = WordSegmenterModel.pretrained()
  .setInputCols("document")
  .setOutputCol("token")

val pipeline = new Pipeline().setStages(Array(
  documentAssembler,
  wordSegmenter
))

val data = Seq("然而，這樣的處理也衍生了一些問題。").toDF("text")
val result = pipeline.fit(data).transform(data)

result.select("token.result").show(false)
+--------------------------------------------------------+
|result                                                  |
+--------------------------------------------------------+
|[然而, ，, 這樣, 的, 處理, 也, 衍生, 了, 一些, 問題, 。    ]|
+--------------------------------------------------------+

{%- endcapture -%}

{%- capture model_api_link -%}
[WordSegmenterModel](https://nlp.johnsnowlabs.com/api/com/johnsnowlabs/nlp/annotators/ws/WordSegmenterModel)
{%- endcapture -%}

{%- capture model_python_api_link -%}
[WordSegmenterModel](/api/python/reference/autosummary/python/sparknlp/annotator/ws/word_segmenter/index.html#sparknlp.annotator.ws.word_segmenter.WordSegmenterModel)
{%- endcapture -%}

{%- capture model_source_link -%}
[WordSegmenterModel](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/ws/WordSegmenterModel.scala)
{%- endcapture -%}

{%- capture approach_description -%}
Trains a WordSegmenter which tokenizes non-english or non-whitespace separated texts.

Many languages are not whitespace separated and their sentences are a concatenation of many symbols, like Korean,
Japanese or Chinese. Without understanding the language, splitting the words into their corresponding tokens is
impossible. The WordSegmenter is trained to understand these languages and split them into semantically correct parts.

For instantiated/pretrained models, see WordSegmenterModel.

To train your own model, a training dataset consisting of
[Part-Of-Speech tags](https://en.wikipedia.org/wiki/Part-of-speech_tagging) is required. The data has to be loaded
into a dataframe, where the column is an [Annotation](https://nlp.johnsnowlabs.com/api/com/johnsnowlabs/nlp/Annotation) of type `"POS"`. This can be
set with `setPosColumn`.

**Tip**: The helper class [POS](/docs/en/training#pos-dataset) might be useful to read training data into data frames.

For extended examples of usage, see the [Spark NLP Workshop](https://github.com/JohnSnowLabs/spark-nlp-workshop/tree/master/jupyter/annotation/chinese/word_segmentation)
and the [WordSegmenterTest](https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/annotators/WordSegmenterTest.scala).
{%- endcapture -%}

{%- capture approach_input_anno -%}
DOCUMENT
{%- endcapture -%}

{%- capture approach_output_anno -%}
TOKEN
{%- endcapture -%}

{%- capture approach_python_example -%}
# In this example, `"chinese_train.utf8"` is in the form of
#
# 十|LL 四|RR 不|LL 是|RR 四|LL 十|RR
#
# and is loaded with the `POS` class to create a dataframe of `"POS"` type Annotations.

import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *
from sparknlp.training import *
from pyspark.ml import Pipeline

documentAssembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")

wordSegmenter = WordSegmenterApproach() \
    .setInputCols(["document"]) \
    .setOutputCol("token") \
    .setPosColumn("tags") \
    .setNIterations(5)

pipeline = Pipeline().setStages([
    documentAssembler,
    wordSegmenter
])

trainingDataSet = POS().readDataset(
    spark,
    "src/test/resources/word-segmenter/chinese_train.utf8"
)

pipelineModel = pipeline.fit(trainingDataSet)

{%- endcapture -%}

{%- capture approach_scala_example -%}
// In this example, `"chinese_train.utf8"` is in the form of
//
// 十|LL 四|RR 不|LL 是|RR 四|LL 十|RR
//
// and is loaded with the `POS` class to create a dataframe of `"POS"` type Annotations.

import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.annotators.ws.WordSegmenterApproach
import com.johnsnowlabs.nlp.training.POS
import org.apache.spark.ml.Pipeline

val documentAssembler = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val wordSegmenter = new WordSegmenterApproach()
  .setInputCols("document")
  .setOutputCol("token")
  .setPosColumn("tags")
  .setNIterations(5)

val pipeline = new Pipeline().setStages(Array(
  documentAssembler,
  wordSegmenter
))

val trainingDataSet = POS().readDataset(
  ResourceHelper.spark,
  "src/test/resources/word-segmenter/chinese_train.utf8"
)

val pipelineModel = pipeline.fit(trainingDataSet)

{%- endcapture -%}

{%- capture approach_api_link -%}
[WordSegmenterApproach](https://nlp.johnsnowlabs.com/api/com/johnsnowlabs/nlp/annotators/ws/WordSegmenterApproach)
{%- endcapture -%}

{%- capture approach_python_api_link -%}
[WordSegmenterApproach](/api/python/reference/autosummary/python/sparknlp/annotator/ws/word_segmenter/index.html#sparknlp.annotator.ws.word_segmenter.WordSegmenterApproach)
{%- endcapture -%}

{%- capture approach_source_link -%}
[WordSegmenterApproach](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/ws/WordSegmenterApproach.scala)
{%- endcapture -%}


{% include templates/approach_model_template.md
title=title
model_description=model_description
model_input_anno=model_input_anno
model_output_anno=model_output_anno
model_python_api_link=model_python_api_link
model_api_link=model_api_link
model_source_link=model_source_link
approach_description=approach_description
approach_input_anno=approach_input_anno
approach_output_anno=approach_output_anno
approach_python_example=approach_python_example
approach_scala_example=approach_scala_example
approach_python_api_link=approach_python_api_link
approach_api_link=approach_api_link
approach_source_link=approach_source_link
%}
