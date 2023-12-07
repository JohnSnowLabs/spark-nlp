{%- capture title -%}
DocumentTokenSplitter
{%- endcapture -%}

{%- capture description -%}
Annotator that splits large documents into smaller documents based on the number of tokens in
the text.

Currently, DocumentTokenSplitter splits the text by whitespaces to create the tokens. The
number of these tokens will then be used as a measure of the text length. In the future, other
tokenization techniques will be supported.

For example, given 3 tokens and overlap 1:

```python
"He was, I take it, the most perfect reasoning and observing machine that the world has seen."

["He was, I", "I take it,", "it, the most", "most perfect reasoning", "reasoning and observing", "observing machine that", "that the world", "world has seen."]
```

Additionally, you can set

- whether to trim whitespaces with setTrimWhitespace
- whether to explode the splits to individual rows with setExplodeSplits

For extended examples of usage, see the
[DocumentTokenSplitterTest](https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/annotators/DocumentTokenSplitterTest.scala).
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
   "sherlockholmes.txt",
   wholetext=True
).toDF("text")

documentAssembler = DocumentAssembler().setInputCol("text")

textSplitter = DocumentTokenSplitter() \
    .setInputCols(["document"]) \
    .setOutputCol("splits") \
    .setNumTokens(512) \
    .setTokenOverlap(10) \
    .setExplodeSplits(True)

pipeline = Pipeline().setStages([documentAssembler, textSplitter])

result = pipeline.fit(textDF).transform(textDF)
result.selectExpr(
      "splits.result as result",
      "splits[0].begin as begin",
      "splits[0].end as end",
      "splits[0].end - splits[0].begin as length",
      "splits[0].metadata.numTokens as tokens") \
    .show(8, truncate = 80)
+--------------------------------------------------------------------------------+-----+-----+------+------+
|                                                                          result|begin|  end|length|tokens|
+--------------------------------------------------------------------------------+-----+-----+------+------+
|[ Project Gutenberg's The Adventures of Sherlock Holmes, by Arthur Conan Doyl...|    0| 3018|  3018|   512|
|[study of crime, and occupied his\nimmense faculties and extraordinary powers...| 2950| 5707|  2757|   512|
|[but as I have changed my clothes I can't imagine how you\ndeduce it. As to M...| 5659| 8483|  2824|   512|
|[quarters received. Be in your chamber then at that hour, and do\nnot take it...| 8427|11241|  2814|   512|
|[a pity\nto miss it."\n\n"But your client--"\n\n"Never mind him. I may want y...|11188|13970|  2782|   512|
|[person who employs me wishes his agent to be unknown to\nyou, and I may conf...|13918|16898|  2980|   512|
|[letters back."\n\n"Precisely so. But how--"\n\n"Was there a secret marriage?...|16836|19744|  2908|   512|
|[seven hundred in\nnotes," he said.\n\nHolmes scribbled a receipt upon a shee...|19683|22551|  2868|   512|
+--------------------------------------------------------------------------------+-----+-----+------+------+
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
val textSplitter = new DocumentTokenSplitter()
  .setInputCols("document")
  .setOutputCol("splits")
  .setNumTokens(512)
  .setTokenOverlap(10)
  .setExplodeSplits(true)

val pipeline = new Pipeline().setStages(Array(documentAssembler, textSplitter))
val result = pipeline.fit(textDF).transform(textDF)

result
  .selectExpr(
    "splits.result as result",
    "splits[0].begin as begin",
    "splits[0].end as end",
    "splits[0].end - splits[0].begin as length",
    "splits[0].metadata.numTokens as tokens")
  .show(8, truncate = 80)
+--------------------------------------------------------------------------------+-----+-----+------+------+
|                                                                          result|begin|  end|length|tokens|
+--------------------------------------------------------------------------------+-----+-----+------+------+
|[ Project Gutenberg's The Adventures of Sherlock Holmes, by Arthur Conan Doyl...|    0| 3018|  3018|   512|
|[study of crime, and occupied his\nimmense faculties and extraordinary powers...| 2950| 5707|  2757|   512|
|[but as I have changed my clothes I can't imagine how you\ndeduce it. As to M...| 5659| 8483|  2824|   512|
|[quarters received. Be in your chamber then at that hour, and do\nnot take it...| 8427|11241|  2814|   512|
|[a pity\nto miss it."\n\n"But your client--"\n\n"Never mind him. I may want y...|11188|13970|  2782|   512|
|[person who employs me wishes his agent to be unknown to\nyou, and I may conf...|13918|16898|  2980|   512|
|[letters back."\n\n"Precisely so. But how--"\n\n"Was there a secret marriage?...|16836|19744|  2908|   512|
|[seven hundred in\nnotes," he said.\n\nHolmes scribbled a receipt upon a shee...|19683|22551|  2868|   512|
+--------------------------------------------------------------------------------+-----+-----+------+------+

{%- endcapture -%}

{%- capture api_link -%}
[DocumentTokenSplitter](/api/com/johnsnowlabs/nlp/annotators/DocumentTokenSplitter)
{%- endcapture -%}

{%- capture python_api_link -%}
[DocumentTokenSplitter](/api/python/reference/autosummary/sparknlp/annotator/document_token_splitter/index.html#sparknlp.annotator.document_token_splitter.DocumentTokenSplitter)
{%- endcapture -%}

{%- capture source_link -%}
[DocumentTokenSplitter](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/DocumentTokenSplitter.scala)
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