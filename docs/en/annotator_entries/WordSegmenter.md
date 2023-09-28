{%- capture title -%}
WordSegmenter
{%- endcapture -%}

{%- capture model_description -%}
WordSegmenter which tokenizes non-english or non-whitespace separated texts.

Many languages are not whitespace separated and their sentences are a concatenation of many
symbols, like Korean, Japanese or Chinese. Without understanding the language, splitting the
words into their corresponding tokens is impossible. The WordSegmenter is trained to
understand these languages and plit them into semantically correct parts.

This annotator is based on the paper
[Chinese Word Segmentation as Character Tagging](https://aclanthology.org/O03-4002.pdf). Word
segmentation is treated as a tagging problem. Each character is be tagged as on of four
different labels: LL (left boundary), RR (right boundary), MM (middle) and LR (word by
itself). The label depends on the position of the word in the sentence. LL tagged words will
combine with the word on the right. Likewise, RR tagged words combine with words on the left.
MM tagged words are treated as the middle of the word and combine with either side. LR tagged
words are words by themselves.

Example (from [1], Example 3(a) (raw), 3(b) (tagged), 3(c) (translation)):

- 上海 计划 到 本 世纪 末 实现 人均 国内 生产 总值 五千 美元
- 上/LL 海/RR 计/LL 划/RR 到/LR 本/LR 世/LL 纪/RR 末/LR 实/LL 现/RR 人/LL 均/RR 国/LL 内/RR 生/LL 产/RR 总/LL
  值/RR 五/LL 千/RR 美/LL 元/RR
- Shanghai plans to reach the goal of 5,000 dollars in per capita GDP by the end of the
  century.

This is the instantiated model of the WordSegmenterApproach. For training your own model,
please see the documentation of that class.

Pretrained models can be loaded with `pretrained` of the companion object:

```scala
val wordSegmenter = WordSegmenterModel.pretrained()
  .setInputCols("document")
  .setOutputCol("words_segmented")
```

The default model is `"wordseg_pku"`, default language is `"zh"`, if no values are provided.
For available pretrained models please see the
[Models Hub](https://sparknlp.org/models?task=Word+Segmentation).

For extended examples of usage, see the [Examples](https://github.com/JohnSnowLabs/spark-nlp/blob/master/examples/python/annotation/text/chinese/word_segmentation/words_segmenter_demo.ipynb)
and the [WordSegmenterTest](https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/annotators/WordSegmenterTest.scala).

**References:**

- [[1]](https://aclanthology.org/O03-4002.pdf) Xue, Nianwen. “Chinese Word Segmentation as
  Character Tagging.” International Journal of Computational Linguistics & Chinese Language
  Processing, Volume 8, Number 1, February 2003: Special Issue on Word Formation and Chinese
  Language Processing, 2003, pp. 29-48. ACLWeb, https://aclanthology.org/O03-4002.
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
[WordSegmenterModel](/api/com/johnsnowlabs/nlp/annotators/ws/WordSegmenterModel)
{%- endcapture -%}

{%- capture model_python_api_link -%}
[WordSegmenterModel](/api/python/reference/autosummary/sparknlp/annotator/ws/word_segmenter/index.html#sparknlp.annotator.ws.word_segmenter.WordSegmenterModel)
{%- endcapture -%}

{%- capture model_source_link -%}
[WordSegmenterModel](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/ws/WordSegmenterModel.scala)
{%- endcapture -%}

{%- capture approach_description -%}
Trains a WordSegmenter which tokenizes non-english or non-whitespace separated texts.

Many languages are not whitespace separated and their sentences are a concatenation of many
symbols, like Korean, Japanese or Chinese. Without understanding the language, splitting the
words into their corresponding tokens is impossible. The WordSegmenter is trained to
understand these languages and split them into semantically correct parts.

This annotator is based on the paper Chinese Word Segmentation as Character Tagging [1]. Word
segmentation is treated as a tagging problem. Each character is be tagged as on of four
different labels: LL (left boundary), RR (right boundary), MM (middle) and LR (word by
itself). The label depends on the position of the word in the sentence. LL tagged words will
combine with the word on the right. Likewise, RR tagged words combine with words on the left.
MM tagged words are treated as the middle of the word and combine with either side. LR tagged
words are words by themselves.

Example (from [1], Example 3(a) (raw), 3(b) (tagged), 3(c) (translation)):

- 上海 计划 到 本 世纪 末 实现 人均 国内 生产 总值 五千 美元
- 上/LL 海/RR 计/LL 划/RR 到/LR 本/LR 世/LL 纪/RR 末/LR 实/LL 现/RR 人/LL 均/RR 国/LL 内/RR 生/LL 产/RR 总/LL
  值/RR 五/LL 千/RR 美/LL 元/RR
- Shanghai plans to reach the goal of 5,000 dollars in per capita GDP by the end of the
  century.

For instantiated/pretrained models, see WordSegmenterModel.

To train your own model, a training dataset consisting of
[Part-Of-Speech tags](https://en.wikipedia.org/wiki/Part-of-speech_tagging) is required. The
data has to be loaded into a dataframe, where the column is an
[Annotation](/api/com/johnsnowlabs/nlp/Annotation) of type `"POS"`. This can be set with
`setPosColumn`.

**Tip**: The helper class [POS](/api/com/johnsnowlabs/nlp/training/POS) might be useful to read
training data into data frames.

For extended examples of usage, see the [Examples](https://github.com/JohnSnowLabs/spark-nlp/blob/master/examples/python/annotation/text/chinese/word_segmentation)
and the [WordSegmenterTest](https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/annotators/WordSegmenterTest.scala).

**References:**

- [[1]](https://aclanthology.org/O03-4002.pdf) Xue, Nianwen. “Chinese Word Segmentation as
  Character Tagging.” International Journal of Computational Linguistics & Chinese Language
  Processing, Volume 8, Number 1, February 2003: Special Issue on Word Formation and Chinese
  Language Processing, 2003, pp. 29-48. ACLWeb, https://aclanthology.org/O03-4002.
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
[WordSegmenterApproach](/api/com/johnsnowlabs/nlp/annotators/ws/WordSegmenterApproach)
{%- endcapture -%}

{%- capture approach_python_api_link -%}
[WordSegmenterApproach](/api/python/reference/autosummary/sparknlp/annotator/ws/word_segmenter/index.html#sparknlp.annotator.ws.word_segmenter.WordSegmenterApproach)
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
