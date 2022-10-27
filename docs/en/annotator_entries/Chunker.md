{%- capture title -%}
Chunker
{%- endcapture -%}

{%- capture description -%}
This annotator matches a pattern of part-of-speech tags in order to return meaningful phrases from document.
Extracted part-of-speech tags are mapped onto the sentence, which can then be parsed by regular expressions.
The part-of-speech tags are wrapped by angle brackets `<>` to be easily distinguishable in the text itself.
This example sentence will result in the form:
```
"Peter Pipers employees are picking pecks of pickled peppers."
"<NNP><NNP><NNS><VBP><VBG><NNS><IN><JJ><NNS><.>"
```
To then extract these tags, `regexParsers` need to be set with e.g.:
```
val chunker = new Chunker()
  .setInputCols("sentence", "pos")
  .setOutputCol("chunk")
  .setRegexParsers(Array("<NNP>+", "<NNS>+"))
```
When defining the regular expressions, tags enclosed in angle brackets are treated as groups, so here specifically
`"<NNP>+"` means 1 or more nouns in succession. Additional patterns can also be set with `addRegexParsers`.

For more extended examples see the [Spark NLP Workshop](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Public/3.SparkNLP_Pretrained_Models.ipynb))
and the  [ChunkerTestSpec](https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/annotators/ChunkerTestSpec.scala).
{%- endcapture -%}

{%- capture input_anno -%}
DOCUMENT, POS
{%- endcapture -%}

{%- capture output_anno -%}
CHUNK
{%- endcapture -%}

{%- capture python_example -%}
import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline

documentAssembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")

sentence = SentenceDetector() \
    .setInputCols("document") \
    .setOutputCol("sentence")

tokenizer = Tokenizer() \
    .setInputCols(["sentence"]) \
    .setOutputCol("token")

POSTag = PerceptronModel.pretrained() \
    .setInputCols("document", "token") \
    .setOutputCol("pos")

chunker = Chunker() \
    .setInputCols("sentence", "pos") \
    .setOutputCol("chunk") \
    .setRegexParsers(["<NNP>+", "<NNS>+"])

pipeline = Pipeline() \
    .setStages([
      documentAssembler,
      sentence,
      tokenizer,
      POSTag,
      chunker
    ])

data = spark.createDataFrame([["Peter Pipers employees are picking pecks of pickled peppers."]]).toDF("text")
result = pipeline.fit(data).transform(data)

result.selectExpr("explode(chunk) as result").show(truncate=False)
+-------------------------------------------------------------+
|result                                                       |
+-------------------------------------------------------------+
|[chunk, 0, 11, Peter Pipers, [sentence -> 0, chunk -> 0], []]|
|[chunk, 13, 21, employees, [sentence -> 0, chunk -> 1], []]  |
|[chunk, 35, 39, pecks, [sentence -> 0, chunk -> 2], []]      |
|[chunk, 52, 58, peppers, [sentence -> 0, chunk -> 3], []]    |
+-------------------------------------------------------------+

{%- endcapture -%}

{%- capture scala_example -%}
import spark.implicits._
import com.johnsnowlabs.nlp.DocumentAssembler
import com.johnsnowlabs.nlp.annotators.{Chunker, Tokenizer}
import com.johnsnowlabs.nlp.annotators.pos.perceptron.PerceptronModel
import com.johnsnowlabs.nlp.annotators.sbd.pragmatic.SentenceDetector
import org.apache.spark.ml.Pipeline

val documentAssembler = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val sentence = new SentenceDetector()
  .setInputCols("document")
  .setOutputCol("sentence")

val tokenizer = new Tokenizer()
  .setInputCols(Array("sentence"))
  .setOutputCol("token")

val POSTag = PerceptronModel.pretrained()
  .setInputCols("document", "token")
  .setOutputCol("pos")

val chunker = new Chunker()
  .setInputCols("sentence", "pos")
  .setOutputCol("chunk")
  .setRegexParsers(Array("<NNP>+", "<NNS>+"))

val pipeline = new Pipeline()
  .setStages(Array(
    documentAssembler,
    sentence,
    tokenizer,
    POSTag,
    chunker
  ))

val data = Seq("Peter Pipers employees are picking pecks of pickled peppers.").toDF("text")
val result = pipeline.fit(data).transform(data)

result.selectExpr("explode(chunk) as result").show(false)
+-------------------------------------------------------------+
|result                                                       |
+-------------------------------------------------------------+
|[chunk, 0, 11, Peter Pipers, [sentence -> 0, chunk -> 0], []]|
|[chunk, 13, 21, employees, [sentence -> 0, chunk -> 1], []]  |
|[chunk, 35, 39, pecks, [sentence -> 0, chunk -> 2], []]      |
|[chunk, 52, 58, peppers, [sentence -> 0, chunk -> 3], []]    |
+-------------------------------------------------------------+

{%- endcapture -%}

{%- capture api_link -%}
[Chunker](https://nlp.johnsnowlabs.com/api/com/johnsnowlabs/nlp/annotators/Chunker)
{%- endcapture -%}

{%- capture python_api_link -%}
[Chunker](/api/python/reference/autosummary/python/sparknlp/annotator/chunker/index.html#sparknlp.annotator.chunker.Chunker)
{%- endcapture -%}

{%- capture source_link -%}
[Chunker](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/Chunker.scala)
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