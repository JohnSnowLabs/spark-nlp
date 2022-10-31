{%- capture title -%}
ChunkFilterer
{%- endcapture -%}

{%- capture model -%}
model
{%- endcapture -%}

{%- capture model_description -%}
Filters entities coming from CHUNK annotations. Filters can be set via a white list of terms or a regular expression.
White list criteria is enabled by default. To use regex, `criteria` has to be set to `regex`.
{%- endcapture -%}

{%- capture model_input_anno -%}
DOCUMENT,CHUNK
{%- endcapture -%}

{%- capture model_output_anno -%}
CHUNK
{%- endcapture -%}

{%- capture model_python_medical -%}
from johnsnowlabs import *
# Filtering POS tags
# First pipeline stages to extract the POS tags are defined
data = spark.createDataFrame([["Has a past history of gastroenteritis and stomach pain, however patient ..."]]).toDF("text")
docAssembler = nlp.DocumentAssembler().setInputCol("text").setOutputCol("document")
sentenceDetector = nlp.SentenceDetector().setInputCols(["document"]).setOutputCol("sentence")
tokenizer = nlp.Tokenizer().setInputCols(["sentence"]).setOutputCol("token")

posTagger = nlp.PerceptronModel.pretrained() \
  .setInputCols(["sentence", "token"]) \
  .setOutputCol("pos")

chunker = nlp.Chunker() \
  .setInputCols(["pos", "sentence"]) \
  .setOutputCol("chunk") \
  .setRegexParsers(["(<NN>)+"])

# Then the chunks can be filtered via a white list. Here only terms with "gastroenteritis" remain.
chunkerFilter = medical.ChunkFilterer() \
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

{%- capture model_python_legal -%}
from johnsnowlabs import *
# Filtering POS tags
# First pipeline stages to extract the POS tags are defined

docAssembler = nlp.DocumentAssembler().setInputCol("text").setOutputCol("document")
sentenceDetector = nlp.SentenceDetector().setInputCols(["document"]).setOutputCol("sentence")
tokenizer = nlp.Tokenizer().setInputCols(["sentence"]).setOutputCol("token")

posTagger = nlp.PerceptronModel.pretrained() \
  .setInputCols(["sentence", "token"]) \
  .setOutputCol("pos")

chunker = nlp.Chunker() \
  .setInputCols(["pos", "sentence"]) \
  .setOutputCol("chunk") \
  .setRegexParsers(["(<NN>)+"])

# Then the chunks can be filtered via a white list. Here only terms with "gastroenteritis" remain.
chunkerFilter = legal.ChunkFilterer() \
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
{%- endcapture -%}


{%- capture model_python_finance -%}
from johnsnowlabs import *
# Filtering POS tags
# First pipeline stages to extract the POS tags are defined

docAssembler = nlp.DocumentAssembler().setInputCol("text").setOutputCol("document")
sentenceDetector = nlp.SentenceDetector().setInputCols(["document"]).setOutputCol("sentence")
tokenizer = nlp.Tokenizer().setInputCols(["sentence"]).setOutputCol("token")

posTagger = nlp.PerceptronModel.pretrained() \
  .setInputCols(["sentence", "token"]) \
  .setOutputCol("pos")

chunker = nlp.Chunker() \
  .setInputCols(["pos", "sentence"]) \
  .setOutputCol("chunk") \
  .setRegexParsers(["(<NN>)+"])

# Then the chunks can be filtered via a white list. Here only terms with "gastroenteritis" remain.
chunkerFilter = finance.ChunkFilterer() \
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
{%- endcapture -%}


{%- capture model_scala_medical -%}
from johnsnowlabs import * 
// Filtering POS tags
// First pipeline stages to extract the POS tags are defined
val data = Seq("Has a past history of gastroenteritis and stomach pain, however patient ...").toDF("text")
val docAssembler = new nlp.DocumentAssembler().setInputCol("text").setOutputCol("document")
val sentenceDetector = new nlp.SentenceDetector().setInputCols("document").setOutputCol("sentence")
val tokenizer = new nlp.Tokenizer().setInputCols("sentence").setOutputCol("token")

val posTagger = nlp.PerceptronModel.pretrained()
  .setInputCols(Array("sentence", "token"))
  .setOutputCol("pos")

val chunker = new nlp.Chunker()
  .setInputCols(Array("pos", "sentence"))
  .setOutputCol("chunk")
  .setRegexParsers(Array("(<NN>)+"))

// Then the chunks can be filtered via a white list. Here only terms with "gastroenteritis" remain.
val chunkerFilter = new medical.ChunkFilterer()
  .setInputCols(Array("sentence","chunk"))
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


{%- capture model_scala_legal -%}
from johnsnowlabs import * 

val docAssembler = new nlp.DocumentAssembler().setInputCol("text").setOutputCol("document")
val sentenceDetector = new nlp.SentenceDetector().setInputCols("document").setOutputCol("sentence")
val tokenizer = new nlp.Tokenizer().setInputCols("sentence").setOutputCol("token")

val posTagger = nlp.PerceptronModel.pretrained()
  .setInputCols(Array("sentence", "token"))
  .setOutputCol("pos")

val chunker = new nlp.Chunker()
  .setInputCols(Array("pos", "sentence"))
  .setOutputCol("chunk")
  .setRegexParsers(Array("(<NN>)+"))

// Then the chunks can be filtered via a white list. Here only terms with "gastroenteritis" remain.
val chunkerFilter = new legal.ChunkFilterer()
  .setInputCols(Array("sentence","chunk"))
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
{%- endcapture -%}

{%- capture model_scala_finance -%}
from johnsnowlabs import * 

val docAssembler = new nlp.DocumentAssembler().setInputCol("text").setOutputCol("document")
val sentenceDetector = new nlp.SentenceDetector().setInputCols("document").setOutputCol("sentence")
val tokenizer = new nlp.Tokenizer().setInputCols("sentence").setOutputCol("token")

val posTagger = nlp.PerceptronModel.pretrained()
  .setInputCols(Array("sentence", "token"))
  .setOutputCol("pos")

val chunker = new nlp.Chunker()
  .setInputCols(Array("pos", "sentence"))
  .setOutputCol("chunk")
  .setRegexParsers(Array("(<NN>)+"))

// Then the chunks can be filtered via a white list. Here only terms with "gastroenteritis" remain.
val chunkerFilter = new finance.ChunkFilterer()
  .setInputCols(Array("sentence","chunk"))
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
{%- endcapture -%}


{%- capture model_api_link -%}
[ChunkFilterer](https://nlp.johnsnowlabs.com/licensed/api/com/johnsnowlabs/nlp/annotators/chunker/ChunkFilterer)
{%- endcapture -%}

{% include templates/licensed_approach_model_medical_fin_leg_template.md
title=title
model=model
model_description=model_description
model_input_anno=model_input_anno
model_output_anno=model_output_anno
model_python_medical=model_python_medical
model_python_legal=model_python_legal
model_python_finance=model_python_finance
model_scala_medical=model_scala_medical
model_scala_legal=model_scala_legal
model_scala_finance=model_scala_finance
model_api_link=model_api_link%}