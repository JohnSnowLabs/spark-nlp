{%- capture model -%}
model
{%- endcapture -%}

{%- capture title -%}
AssertionFilterer
{%- endcapture -%}

{%- capture model_description -%}
Filters entities coming from ASSERTION type annotations and returns the CHUNKS.
Filters can be set via a white list on the extracted chunk, the assertion or a regular expression.
White list for assertion is enabled by default. To use chunk white list, `criteria` has to be set to `"isin"`.
For regex, `criteria` has to be set to `"regex"`.
{%- endcapture -%}

{%- capture model_input_anno -%}
DOCUMENT, CHUNK, ASSERTION
{%- endcapture -%}

{%- capture model_output_anno -%}
CHUNK
{%- endcapture -%}


{%- capture model_python_medical -%}
from johnsnowlabs import * 
# To see how the assertions are extracted, see the example for AssertionDLModel.
# Define an extra step where the assertions are filtered
assertionFilterer = medical.AssertionFilterer() \
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

{%- capture model_python_legal -%}
from johnsnowlabs import * 
# To see how the assertions are extracted, see the example for AssertionDLModel.
# Define an extra step where the assertions are filtered
assertionFilterer = legal.AssertionFilterer() \
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
{%- endcapture -%}

{%- capture model_python_finance -%}
from johnsnowlabs import * 
# To see how the assertions are extracted, see the example for AssertionDLModel.
# Define an extra step where the assertions are filtered
assertionFilterer = finance.AssertionFilterer() \
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
{%- endcapture -%}

{%- capture model_scala_medical -%}
from johnsnowlabs import * 

// To see how the assertions are extracted, see the example for
// [[com.johnsnowlabs.nlp.annotators.assertion.dl.AssertionDLModel AssertionDLModel]].
// Define an extra step where the assertions are filtered
val assertionFilterer = new medical.AssertionFilterer()
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

{%- capture model_scala_legal -%}
from johnsnowlabs import * 

// To see how the assertions are extracted, see the example for
// [[com.johnsnowlabs.nlp.annotators.assertion.dl.AssertionDLModel AssertionDLModel]].
// Define an extra step where the assertions are filtered
val assertionFilterer = new legal.AssertionFilterer()
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
{%- endcapture -%}

{%- capture model_scala_finance -%}
from johnsnowlabs import * 

// To see how the assertions are extracted, see the example for
// [[com.johnsnowlabs.nlp.annotators.assertion.dl.AssertionDLModel AssertionDLModel]].
// Define an extra step where the assertions are filtered
val assertionFilterer = new legal.AssertionFilterer()
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
{%- endcapture -%}

{%- capture model_api_link -%}
[AssertionFilterer](https://nlp.johnsnowlabs.com/licensed/api/com/johnsnowlabs/nlp/annotators/chunker/AssertionFilterer)
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