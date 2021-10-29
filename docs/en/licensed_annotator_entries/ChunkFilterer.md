{%- capture title -%}
ChunkFilterer
{%- endcapture -%}

{%- capture description -%}
Filters entities coming from CHUNK annotations. Filters can be set via a white list of terms or a regular expression.
White list criteria is enabled by default. To use regex, `criteria` has to be set to `regex`.
{%- endcapture -%}

{%- capture input_anno -%}
DOCUMENT,CHUNK
{%- endcapture -%}

{%- capture output_anno -%}
CHUNK
{%- endcapture -%}

{%- capture python_example -%}
import sparknlp
from sparknlp.base import *
from sparknlp.common import *
from sparknlp.annotator import *
from sparknlp.training import *
import sparknlp_jsl
from sparknlp_jsl.base import *
from sparknlp_jsl.annotator import *
from pyspark.ml import Pipeline
# Filtering POS tags
# First pipeline stages to extract the POS tags are defined
data = spark.createDataFrame([["Has a past history of gastroenteritis and stomach pain, however patient ..."]]).toDF("text")
docAssembler = DocumentAssembler().setInputCol("text").setOutputCol("document")
sentenceDetector = SentenceDetector().setInputCols(["document"]).setOutputCol("sentence")
tokenizer = Tokenizer().setInputCols(["sentence"]).setOutputCol("token")

posTagger = PerceptronModel.pretrained() \
  .setInputCols(["sentence", "token"]) \
  .setOutputCol("pos")

chunker = Chunker() \
  .setInputCols(["pos", "sentence"]) \
  .setOutputCol("chunk") \
  .setRegexParsers(["(<NN>)+"])

# Then the chunks can be filtered via a white list. Here only terms with "gastroenteritis" remain.
chunkerFilter = ChunkFilterer() \
  .setInputCols(["sentence","chunk"]) \
  .setOutputCol("filtered") \
  .setCriteria("isin") \
  .setWhiteList(["gastroenteritis"])

pipeline = Pipeline(stages=[
  docAssembler,
  sentenceDetector,
  tokenizer,
  posTagger,
  chunker,
  chunkerFilter])

result = pipeline.fit(data).transform(data)
result.selectExpr("explode(chunk)").show(truncate=False)
+---------------------------------------------------------------------------------+
|col                                                                              |
+---------------------------------------------------------------------------------+
|{chunk, 11, 17, history, {sentence -> 0, chunk -> 0}, []}                        |
|{chunk, 22, 36, gastroenteritis, {sentence -> 0, chunk -> 1}, []}                |
|{chunk, 42, 53, stomach pain, {sentence -> 0, chunk -> 2}, []}                   |
|{chunk, 64, 70, patient, {sentence -> 0, chunk -> 3}, []}                        |
|{chunk, 81, 110, stomach pain now.We don't care, {sentence -> 0, chunk -> 4}, []}|
|{chunk, 118, 132, gastroenteritis, {sentence -> 0, chunk -> 5}, []}              |
+---------------------------------------------------------------------------------+

result.selectExpr("explode(filtered)").show(truncate=False)
+-------------------------------------------------------------------+
|col                                                                |
+-------------------------------------------------------------------+
|{chunk, 22, 36, gastroenteritis, {sentence -> 0, chunk -> 1}, []}  |
|{chunk, 118, 132, gastroenteritis, {sentence -> 0, chunk -> 5}, []}|
+-------------------------------------------------------------------+

{%- endcapture -%}

{%- capture scala_example -%}
// Filtering POS tags
// First pipeline stages to extract the POS tags are defined
val data = Seq("Has a past history of gastroenteritis and stomach pain, however patient ...").toDF("text")
val docAssembler = new DocumentAssembler().setInputCol("text").setOutputCol("document")
val sentenceDetector = new SentenceDetector().setInputCols("document").setOutputCol("sentence")
val tokenizer = new Tokenizer().setInputCols("sentence").setOutputCol("token")

val posTagger = PerceptronModel.pretrained()
  .setInputCols("sentence", "token")
  .setOutputCol("pos")

val chunker = new Chunker()
  .setInputCols("pos", "sentence")
  .setOutputCol("chunk")
  .setRegexParsers(Array("(<NN>)+"))

// Then the chunks can be filtered via a white list. Here only terms with "gastroenteritis" remain.
val chunkerFilter = new ChunkFilterer()
  .setInputCols("sentence","chunk")
  .setOutputCol("filtered")
  .setCriteria("isin")
  .setWhiteList("gastroenteritis")

val pipeline = new Pipeline().setStages(Array(
  docAssembler,
  sentenceDetector,
  tokenizer,
  posTagger,
  chunker,
  chunkerFilter))

result.selectExpr("explode(chunk)").show(truncate=false)
+---------------------------------------------------------------------------------+
|col                                                                              |
+---------------------------------------------------------------------------------+
|{chunk, 11, 17, history, {sentence -> 0, chunk -> 0}, []}                        |
|{chunk, 22, 36, gastroenteritis, {sentence -> 0, chunk -> 1}, []}                |
|{chunk, 42, 53, stomach pain, {sentence -> 0, chunk -> 2}, []}                   |
|{chunk, 64, 70, patient, {sentence -> 0, chunk -> 3}, []}                        |
|{chunk, 81, 110, stomach pain now.We don't care, {sentence -> 0, chunk -> 4}, []}|
|{chunk, 118, 132, gastroenteritis, {sentence -> 0, chunk -> 5}, []}              |
+---------------------------------------------------------------------------------+

result.selectExpr("explode(filtered)").show(truncate=false)
+-------------------------------------------------------------------+
|col                                                                |
+-------------------------------------------------------------------+
|{chunk, 22, 36, gastroenteritis, {sentence -> 0, chunk -> 1}, []}  |
|{chunk, 118, 132, gastroenteritis, {sentence -> 0, chunk -> 5}, []}|
+-------------------------------------------------------------------+

{%- endcapture -%}

{%- capture api_link -%}
[ChunkFilterer](https://nlp.johnsnowlabs.com/licensed/api/com/johnsnowlabs/nlp/annotators/chunker/ChunkFilterer)
{%- endcapture -%}

{% include templates/licensed_anno_template.md
title=title
description=description
input_anno=input_anno
output_anno=output_anno
python_example=python_example
scala_example=scala_example
api_link=api_link%}