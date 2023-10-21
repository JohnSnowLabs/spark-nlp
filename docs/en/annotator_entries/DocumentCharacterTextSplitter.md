{%- capture title -%}
DocumentCharacterTextSplitter
{%- endcapture -%}

{%- capture description -%}
Annotator which splits large documents into chunks of roughly given size.

DocumentCharacterTextSplitter takes a list of separators. It takes the separators in order and
splits subtexts if they are over the chunk length, considering optional overlap of the chunks.

For example, given chunk size 20 and overlap 5:

```python
"He was, I take it, the most perfect reasoning and observing machine that the world has seen."

["He was, I take it,", "it, the most", "most perfect", "reasoning and", "and observing", "machine that the", "the world has seen."]
```

Additionally, you can set

- custom patterns with setSplitPatterns
- whether patterns should be interpreted as regex with setPatternsAreRegex
- whether to keep the separators with setKeepSeparators
- whether to trim whitespaces with setTrimWhitespace
- whether to explode the splits to individual rows with setExplodeSplits

For extended examples of usage, see the
[DocumentCharacterTextSplitterTest](https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/annotators/DocumentCharacterTextSplitterTest.scala).
{%- endcapture -%}

{%- capture input_anno -%}
DOCUMENT
{%- endcapture -%}

{%- capture output_anno -%}
DOCUMENT
{%- endcapture -%}

{%- capture python_example -%}
import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline

textDF = spark.read.text(
   "/home/ducha/Workspace/scala/spark-nlp/src/test/resources/spell/sherlockholmes.txt",
   wholetext=True
).toDF("text")

documentAssembler = DocumentAssembler().setInputCol("text")

textSplitter = DocumentCharacterTextSplitter() \
    .setInputCols(["document"]) \
    .setOutputCol("splits") \
    .setChunkSize(20000) \
    .setChunkOverlap(200) \
    .setExplodeSplits(True)

pipeline = Pipeline().setStages([documentAssembler, textSplitter])
result = pipeline.fit(textDF).transform(textDF)
result.selectExpr(
      "splits.result",
      "splits[0].begin",
      "splits[0].end",
      "splits[0].end - splits[0].begin as length") \
    .show(8, truncate = 80)
+--------------------------------------------------------------------------------+---------------+-------------+------+
|                                                                          result|splits[0].begin|splits[0].end|length|
+--------------------------------------------------------------------------------+---------------+-------------+------+
|[ Project Gutenberg's The Adventures of Sherlock Holmes, by Arthur Conan Doyl...|              0|        19994| 19994|
|["And Mademoiselle's address?" he asked.\n\n"Is Briony Lodge, Serpentine Aven...|          19798|        39395| 19597|
|["How did that help you?"\n\n"It was all-important. When a woman thinks that ...|          39371|        59242| 19871|
|["'But,' said I, 'there would be millions of red-headed men who\nwould apply....|          59166|        77833| 18667|
|[My friend was an enthusiastic musician, being himself not only a\nvery capab...|          77835|        97769| 19934|
|["And yet I am not convinced of it," I answered. "The cases which\ncome to li...|          97771|       117248| 19477|
|["Well, she had a slate-coloured, broad-brimmed straw hat, with a\nfeather of...|         117250|       137242| 19992|
|["That sounds a little paradoxical."\n\n"But it is profoundly True. Singulari...|         137244|       157171| 19927|
+--------------------------------------------------------------------------------+---------------+-------------+------+
{%- endcapture -%}

{%- capture scala_example -%}
import com.johnsnowlabs.nlp.annotator._
import com.johnsnowlabs.nlp.DocumentAssembler
import org.apache.spark.ml.Pipeline

val textDF =
  spark.read
    .option("wholetext", "true")
    .text("src/test/resources/spell/sherlockholmes.txt")
    .toDF("text")

val documentAssembler = new DocumentAssembler().setInputCol("text")
val textSplitter = new DocumentCharacterTextSplitter()
  .setInputCols("document")
  .setOutputCol("splits")
  .setChunkSize(20000)
  .setChunkOverlap(200)
  .setExplodeSplits(true)

val pipeline = new Pipeline().setStages(Array(documentAssembler, textSplitter))
val result = pipeline.fit(textDF).transform(textDF)

result
  .selectExpr(
    "splits.result",
    "splits[0].begin",
    "splits[0].end",
    "splits[0].end - splits[0].begin as length")
  .show(8, truncate = 80)
+--------------------------------------------------------------------------------+---------------+-------------+------+
|                                                                          result|splits[0].begin|splits[0].end|length|
+--------------------------------------------------------------------------------+---------------+-------------+------+
|[ Project Gutenberg's The Adventures of Sherlock Holmes, by Arthur Conan Doyl...|              0|        19994| 19994|
|["And Mademoiselle's address?" he asked.\n\n"Is Briony Lodge, Serpentine Aven...|          19798|        39395| 19597|
|["How did that help you?"\n\n"It was all-important. When a woman thinks that ...|          39371|        59242| 19871|
|["'But,' said I, 'there would be millions of red-headed men who\nwould apply....|          59166|        77833| 18667|
|[My friend was an enthusiastic musician, being himself not only a\nvery capab...|          77835|        97769| 19934|
|["And yet I am not convinced of it," I answered. "The cases which\ncome to li...|          97771|       117248| 19477|
|["Well, she had a slate-coloured, broad-brimmed straw hat, with a\nfeather of...|         117250|       137242| 19992|
|["That sounds a little paradoxical."\n\n"But it is profoundly true. Singulari...|         137244|       157171| 19927|
+--------------------------------------------------------------------------------+---------------+-------------+------+

{%- endcapture -%}

{%- capture api_link -%}
[DocumentCharacterTextSplitter](/api/com/johnsnowlabs/nlp/annotators/DocumentCharacterTextSplitter)
{%- endcapture -%}

{%- capture python_api_link -%}
[DocumentCharacterTextSplitter](/api/python/reference/autosummary/sparknlp/annotator/document_character_text_splitter/index.html#sparknlp.annotator.document_character_text_splitter.DocumentCharacterTextSplitter)
{%- endcapture -%}

{%- capture source_link -%}
[DocumentCharacterTextSplitter](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/DocumentCharacterTextSplitter.scala)
{%- endcapture -%}

{% include templates/anno_template.md
title=title
description=description
input_anno=input_anno
output_anno=output_anno
python_example=python_example
scala_example=scala_example
api_link=api_link
python_api_link=python_api_link
source_link=source_link
%}