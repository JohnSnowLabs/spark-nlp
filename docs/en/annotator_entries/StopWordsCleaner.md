{%- capture title -%}
StopWordsCleaner
{%- endcapture -%}

{%- capture description -%}
This annotator takes a sequence of strings (e.g. the output of a Tokenizer, Normalizer, Lemmatizer, and Stemmer)
and drops all the stop words from the input sequences.

By default, it uses stop words from MLlibs
[StopWordsRemover](https://spark.apache.org/docs/latest/ml-features#stopwordsremover).
Stop words can also be defined by explicitly setting them with `setStopWords(value: Array[String])` or loaded from
pretrained models using `pretrained` of its companion object.
```
val stopWords = StopWordsCleaner.pretrained()
  .setInputCols("token")
  .setOutputCol("cleanTokens")
  .setCaseSensitive(false)
// will load the default pretrained model `"stopwords_en"`.
```
For available pretrained models please see the [Models Hub](https://nlp.johnsnowlabs.com/models?task=Stop+Words+Removal).

For extended examples of usage, see the [Spark NLP Workshop](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Public/2.Text_Preprocessing_with_SparkNLP_Annotators_Transformers.ipynb)
and [StopWordsCleanerTestSpec](https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/annotators/StopWordsCleanerTestSpec.scala).


> **NOTE:**
> If you need to `setStopWords` from a text file, you can first read and convert it into an array of string as follows.

<div class="tabs-box tabs-new" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
# your stop words text file, each line is one stop word
stopwords = sc.textFile("/tmp/stopwords/english.txt").collect()

# simply use it in StopWordsCleaner
stopWordsCleaner = StopWordsCleaner()\
      .setInputCols("token")\
      .setOutputCol("cleanTokens")\
      .setStopWords(stopwords)\
      .setCaseSensitive(False)

# or you can use pretrained models for StopWordsCleaner
stopWordsCleaner = StopWordsCleaner.pretrained()
      .setInputCols("token")\
      .setOutputCol("cleanTokens")\
      .setCaseSensitive(False)

```

```scala
// your stop words text file, each line is one stop word
val stopwords = sc.textFile("/tmp/stopwords/english.txt").collect()

// simply use it in StopWordsCleaner
val stopWordsCleaner = new StopWordsCleaner()
      .setInputCols("token")
      .setOutputCol("cleanTokens")
      .setStopWords(stopwords)
      .setCaseSensitive(false)

// or you can use pretrained models for StopWordsCleaner
val stopWordsCleaner = StopWordsCleaner.pretrained()
      .setInputCols("token")
      .setOutputCol("cleanTokens")
      .setCaseSensitive(false)      
```

{%- endcapture -%}

{%- capture input_anno -%}
TOKEN
{%- endcapture -%}

{%- capture output_anno -%}
TOKEN
{%- endcapture -%}

{%- capture python_example -%}
import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline

documentAssembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")

sentenceDetector = SentenceDetector() \
    .setInputCols(["document"]) \
    .setOutputCol("sentence")

tokenizer = Tokenizer() \
    .setInputCols(["sentence"]) \
    .setOutputCol("token")

stopWords = StopWordsCleaner() \
    .setInputCols(["token"]) \
    .setOutputCol("cleanTokens") \
    .setCaseSensitive(False)

pipeline = Pipeline().setStages([
      documentAssembler,
      sentenceDetector,
      tokenizer,
      stopWords
    ])

data = spark.createDataFrame([
    ["This is my first sentence. This is my second."],
    ["This is my third sentence. This is my forth."]
]).toDF("text")
result = pipeline.fit(data).transform(data)

result.selectExpr("cleanTokens.result").show(truncate=False)
+-------------------------------+
|result                         |
+-------------------------------+
|[first, sentence, ., second, .]|
|[third, sentence, ., forth, .] |
+-------------------------------+

{%- endcapture -%}

{%- capture scala_example -%}
import spark.implicits._
import com.johnsnowlabs.nlp.DocumentAssembler
import com.johnsnowlabs.nlp.annotator.Tokenizer
import com.johnsnowlabs.nlp.annotator.SentenceDetector
import com.johnsnowlabs.nlp.annotators.StopWordsCleaner
import org.apache.spark.ml.Pipeline

val documentAssembler = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val sentenceDetector = new SentenceDetector()
  .setInputCols(Array("document"))
  .setOutputCol("sentence")

val tokenizer = new Tokenizer()
  .setInputCols(Array("sentence"))
  .setOutputCol("token")

val stopWords = new StopWordsCleaner()
  .setInputCols("token")
  .setOutputCol("cleanTokens")
  .setCaseSensitive(false)

val pipeline = new Pipeline().setStages(Array(
    documentAssembler,
    sentenceDetector,
    tokenizer,
    stopWords
  ))

val data = Seq(
  "This is my first sentence. This is my second.",
  "This is my third sentence. This is my forth."
).toDF("text")
val result = pipeline.fit(data).transform(data)

result.selectExpr("cleanTokens.result").show(false)
+-------------------------------+
|result                         |
+-------------------------------+
|[first, sentence, ., second, .]|
|[third, sentence, ., forth, .] |
+-------------------------------+

{%- endcapture -%}

{%- capture api_link -%}
[StopWordsCleaner](https://nlp.johnsnowlabs.com/api/com/johnsnowlabs/nlp/annotators/StopWordsCleaner)
{%- endcapture -%}

{%- capture python_api_link -%}
[StopWordsCleaner](/api/python/reference/autosummary/sparknlp/annotator/stop_words_cleaner/index.html#sparknlp.annotator.stop_words_cleaner.StopWordsCleaner)
{%- endcapture -%}

{%- capture source_link -%}
[StopWordsCleaner](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/StopWordsCleaner.scala)
{%- endcapture -%}

{% include templates/anno_template.md
title=title
description=description
input_anno=input_anno
output_anno=output_anno
python_example=python_example
scala_example=scala_example
python_api_link=python_api_link
api_link=api_link
source_link=source_link
%}