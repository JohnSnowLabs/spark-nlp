{%- capture title -%}
AssertionFilterer
{%- endcapture -%}

{%- capture description -%}
Filters entities coming from ASSERTION type annotations and returns the CHUNKS.
Filters can be set via a white list on the extracted chunk, the assertion or a regular expression.
White list for assertion is enabled by default. To use chunk white list, `criteria` has to be set to `"isin"`.
For regex, `criteria` has to be set to `"regex"`.
{%- endcapture -%}

{%- capture input_anno -%}
DOCUMENT, CHUNK, ASSERTION
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
# To see how the assertions are extracted, see the example for AssertionDLModel.
# Define an extra step where the assertions are filtered
assertionFilterer = AssertionFilterer() \
  .setInputCols(["sentence","ner_chunk","assertion"]) \
  .setOutputCol("filtered") \
  .setCriteria("assertion") \
  .setWhiteList(["present"])

assertionPipeline = Pipeline(stages=[
  documentAssembler,
  sentenceDetector,
  tokenizer,
  embeddings,
  nerModel,
  nerConverter,
  clinicalAssertion,
  assertionFilterer
])

assertionModel = assertionPipeline.fit(data)
result = assertionModel.transform(data)


# Show results:

result.selectExpr("ner_chunk.result", "assertion.result").show(3, truncate=False)
+--------------------------------+--------------------------------+
|result                          |result                          |
+--------------------------------+--------------------------------+
|[severe fever, sore throat]     |[present, present]              |
|[stomach pain]                  |[absent]                        |
|[an epidural, PCA, pain control]|[present, present, hypothetical]|
+--------------------------------+--------------------------------+
result.select("filtered.result").show(3, truncate=False)
+---------------------------+
|result                     |
+---------------------------+
|[severe fever, sore throat]|
|[]                         |
|[an epidural, PCA]         |
+---------------------------+

{%- endcapture -%}

{%- capture scala_example -%}
// To see how the assertions are extracted, see the example for
// [[com.johnsnowlabs.nlp.annotators.assertion.dl.AssertionDLModel AssertionDLModel]].
// Define an extra step where the assertions are filtered
val assertionFilterer = new AssertionFilterer()
  .setInputCols("sentence","ner_chunk","assertion")
  .setOutputCol("filtered")
  .setCriteria("assertion")
  .setWhiteList("present")

val assertionPipeline = new Pipeline().setStages(Array(
  documentAssembler,
  sentenceDetector,
  tokenizer,
  embeddings,
  nerModel,
  nerConverter,
  clinicalAssertion,
  assertionFilterer
))

val assertionModel = assertionPipeline.fit(data)
val result = assertionModel.transform(data)

// Show results:
//
// result.selectExpr("ner_chunk.result", "assertion.result").show(3, truncate=false)
// +--------------------------------+--------------------------------+
// |result                          |result                          |
// +--------------------------------+--------------------------------+
// |[severe fever, sore throat]     |[present, present]              |
// |[stomach pain]                  |[absent]                        |
// |[an epidural, PCA, pain control]|[present, present, hypothetical]|
// +--------------------------------+--------------------------------+
// result.select("filtered.result").show(3, truncate=false)
// +---------------------------+
// |result                     |
// +---------------------------+
// |[severe fever, sore throat]|
// |[]                         |
// |[an epidural, PCA]         |
// +---------------------------+
//
{%- endcapture -%}

{%- capture api_link -%}
[AssertionFilterer](https://nlp.johnsnowlabs.com/licensed/api/com/johnsnowlabs/nlp/annotators/chunker/AssertionFilterer)
{%- endcapture -%}

{% include templates/licensed_anno_template.md
title=title
description=description
input_anno=input_anno
output_anno=output_anno
python_example=python_example
scala_example=scala_example
api_link=api_link%}