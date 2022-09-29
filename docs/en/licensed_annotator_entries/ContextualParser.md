{%- capture title -%}
ContextualParser
{%- endcapture -%}

{%- capture approach -%}
approach
{%- endcapture -%}

{%- capture model -%}
model
{%- endcapture -%}

{%- capture model_description -%}
Extracts entity from a document based on user defined rules. Rule matching is based on a RegexMatcher defined in a
JSON file. In this file, regex is defined that you want to match along with the information that will output on
metadata field. To instantiate a model, see ContextualParserApproach and its accompanied example.
{%- endcapture -%}

{%- capture model_input_anno -%}
DOCUMENT, TOKEN
{%- endcapture -%}

{%- capture model_output_anno -%}
CHUNK
{%- endcapture -%}

{%- capture model_api_link -%}
[ContextualParserModel](https://nlp.johnsnowlabs.com/licensed/api/com/johnsnowlabs/nlp/annotators/context/ContextualParserModel)
{%- endcapture -%}

{%- capture approach_description -%}
Creates a model, that extracts entity from a document based on user defined rules.
Rule matching is based on a RegexMatcher defined in a JSON file. It is set through the parameter setJsonPath()
In this JSON file, regex is defined that you want to match along with the information that will output on metadata
field. Additionally, a dictionary can be provided with `setDictionary` to map extracted entities
to a unified representation. The first column of the dictionary file should be the representation with following
columns the possible matches.
{%- endcapture -%}

{%- capture approach_input_anno -%}
DOCUMENT, TOKEN
{%- endcapture -%}

{%- capture approach_output_anno -%}
CHUNK
{%- endcapture -%}

{%- capture approach_python_medical -%}
from johnsnowlabs import *
# An example JSON file `regex_token.json` can look like this:
#
# {
#    "entity": "Stage",
#    "ruleScope": "sentence",
#    "regex": "[cpyrau]?[T][0-9X?][a-z^cpyrau]",
#    "matchScope": "token"
#  }
#
# Which means to extract the stage code on a sentence level.
# An example pipeline could then be defined like this
# Pipeline could then be defined like this
documentAssembler = nlp.DocumentAssembler() \
  .setInputCol("text") \
  .setOutputCol("document")

sentenceDetector = nlp.SentenceDetector() \
  .setInputCols(["document"]) \
  .setOutputCol("sentence")

tokenizer = nlp.Tokenizer() \
  .setInputCols(["sentence"]) \
  .setOutputCol("token")

# Define the parser (json file needs to be provided)
data = spark.createDataFrame([["A patient has liver metastases pT1bN0M0 and the T5 primary site may be colon or... "]]).toDF("text")

contextualParser = medical.ContextualParserApproach() \
  .setInputCols(["sentence", "token"]) \
  .setOutputCol("entity") \
  .setJsonPath("/path/to/regex_token.json") \
  .setCaseSensitive(True) \
  .setContextMatch(False)

pipeline = Pipeline(stages=[
    documentAssembler,
    sentenceDetector,
    tokenizer,
    contextualParser
  ])

result = pipeline.fit(data).transform(data)

# Show Results
result.selectExpr("explode(entity)").show(5, truncate=False)
+-------------------------------------------------------------------------------------------------------------------------+
|col                                                                                                                      |
+-------------------------------------------------------------------------------------------------------------------------+
|{chunk, 32, 39, pT1bN0M0, {field -> Stage, normalized -> , confidenceValue -> 0.13, hits -> regex, sentence -> 0}, []}   |
|{chunk, 49, 50, T5, {field -> Stage, normalized -> , confidenceValue -> 0.13, hits -> regex, sentence -> 0}, []}         |
|{chunk, 148, 156, cT4bcN2M1, {field -> Stage, normalized -> , confidenceValue -> 0.13, hits -> regex, sentence -> 1}, []}|
|{chunk, 189, 194, T?N3M1, {field -> Stage, normalized -> , confidenceValue -> 0.13, hits -> regex, sentence -> 2}, []}   |
|{chunk, 316, 323, pT1bN0M0, {field -> Stage, normalized -> , confidenceValue -> 0.13, hits -> regex, sentence -> 3}, []} |
+-------------------------------------------------------------------------------------------------------------------------+
{%- endcapture -%}

{%- capture approach_scala_medical -%}
from johnsnowlabs import * 
// An example JSON file `regex_token.json` can look like this:
//
// {
//    "entity": "Stage",
//    "ruleScope": "sentence",
//    "regex": "[cpyrau]?[T][0-9X?][a-z^cpyrau]",
//    "matchScope": "token"
//  }
//
// Which means to extract the stage code on a sentence level.
// An example pipeline could then be defined like this
val documentAssembler = new nlp.DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val sentenceDetector = new nlp.SentenceDetector()
  .setInputCols("document")
  .setOutputCol("sentence")

val tokenizer = new nlp.Tokenizer()
  .setInputCols("sentence")
  .setOutputCol("token")

// Define the parser (json file needs to be provided)
val data = Seq("A patient has liver metastases pT1bN0M0 and the T5 primary site may be colon or... ").toDF("text")
val contextualParser = new medical.ContextualParserApproach()
  .setInputCols(Array("sentence", "token"))
  .setOutputCol("entity")
  .setJsonPath("/path/to/regex_token.json")
  .setCaseSensitive(true)
  .setContextMatch(false)
val pipeline = new Pipeline().setStages(Array(
    documentAssembler,
    sentenceDetector,
    tokenizer,
    contextualParser
  ))

val result = pipeline.fit(data).transform(data)

// Show Results
//
// result.selectExpr("explode(entity)").show(5, truncate=false)
// +-------------------------------------------------------------------------------------------------------------------------+
// |col                                                                                                                      |
// +-------------------------------------------------------------------------------------------------------------------------+
// |{chunk, 32, 39, pT1bN0M0, {field -> Stage, normalized -> , confidenceValue -> 0.13, hits -> regex, sentence -> 0}, []}   |
// |{chunk, 49, 50, T5, {field -> Stage, normalized -> , confidenceValue -> 0.13, hits -> regex, sentence -> 0}, []}         |
// |{chunk, 148, 156, cT4bcN2M1, {field -> Stage, normalized -> , confidenceValue -> 0.13, hits -> regex, sentence -> 1}, []}|
// |{chunk, 189, 194, T?N3M1, {field -> Stage, normalized -> , confidenceValue -> 0.13, hits -> regex, sentence -> 2}, []}   |
// |{chunk, 316, 323, pT1bN0M0, {field -> Stage, normalized -> , confidenceValue -> 0.13, hits -> regex, sentence -> 3}, []} |
// +-------------------------------------------------------------------------------------------------------------------------+
//
{%- endcapture -%}

{%- capture approach_api_link -%}
[ContextualParserApproach](https://nlp.johnsnowlabs.com/licensed/api/com/johnsnowlabs/nlp/annotators/context/ContextualParserApproach)
{%- endcapture -%}


{% include templates/licensed_approach_model_medical_fin_leg_template.md
title=title
model=model
approach=approach
model_description=model_description
model_input_anno=model_input_anno
model_output_anno=model_output_anno
model_api_link=model_api_link
approach_description=approach_description
approach_input_anno=approach_input_anno
approach_output_anno=approach_output_anno
approach_python_medical=approach_python_medical
approach_scala_medical=approach_scala_medical
approach_api_link=approach_api_link
%}
