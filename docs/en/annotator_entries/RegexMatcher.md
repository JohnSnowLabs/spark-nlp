{%- capture title -%}
RegexMatcher
{%- endcapture -%}

{%- capture model_description -%}
Instantiated model of the RegexMatcher.
For usage and examples see the documentation of the main class.
{%- endcapture -%}

{%- capture model_input_anno -%}
DOCUMENT
{%- endcapture -%}

{%- capture model_output_anno -%}
CHUNK
{%- endcapture -%}

{%- capture model_api_link -%}
[RegexMatcherModel](https://nlp.johnsnowlabs.com/api/com/johnsnowlabs/nlp/annotators/RegexMatcherModel)
{%- endcapture -%}

{%- capture model_python_api_link -%}
[RegexMatcherModel](/api/python/reference/autosummary/python/sparknlp/annotator/matcher/regex_matcher/index.html#sparknlp.annotator.matcher.regex_matcher.RegexMatcherModel)
{%- endcapture -%}

{%- capture model_source_link -%}
[RegexMatcherModel](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/RegexMatcherModel.scala)
{%- endcapture -%}

{%- capture approach_description -%}
Uses a reference file to match a set of regular expressions and associate them with a provided identifier.

A dictionary of predefined regular expressions must be provided with `setExternalRules`.
The dictionary can be set as a delimited text file.

Pretrained pipelines are available for this module, see [Pipelines](https://nlp.johnsnowlabs.com/docs/en/pipelines).

For extended examples of usage, see the [Spark NLP Workshop](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Public/2.Text_Preprocessing_with_SparkNLP_Annotators_Transformers.ipynb)
and the [RegexMatcherTestSpec](https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/annotators/RegexMatcherTestSpec.scala).
{%- endcapture -%}

{%- capture approach_input_anno -%}
DOCUMENT
{%- endcapture -%}

{%- capture approach_output_anno -%}
CHUNK
{%- endcapture -%}

{%- capture approach_python_example -%}
import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline
# In this example, the `rules.txt` has the form of
#
# the\s\w+, followed by 'the'
# ceremonies, ceremony
#
# where each regex is separated by the identifier by `","`

documentAssembler = DocumentAssembler().setInputCol("text").setOutputCol("document")

sentence = SentenceDetector().setInputCols(["document"]).setOutputCol("sentence")

regexMatcher = RegexMatcher() \
    .setExternalRules("src/test/resources/regex-matcher/rules.txt",  ",") \
    .setInputCols(["sentence"]) \
    .setOutputCol("regex") \
    .setStrategy("MATCH_ALL")

pipeline = Pipeline().setStages([documentAssembler, sentence, regexMatcher])

data = spark.createDataFrame([[
    "My first sentence with the first rule. This is my second sentence with ceremonies rule."
]]).toDF("text")
results = pipeline.fit(data).transform(data)

results.selectExpr("explode(regex) as result").show(truncate=False)
+--------------------------------------------------------------------------------------------+
|result                                                                                      |
+--------------------------------------------------------------------------------------------+
|[chunk, 23, 31, the first, [identifier -> followed by 'the', sentence -> 0, chunk -> 0], []]|
|[chunk, 71, 80, ceremonies, [identifier -> ceremony, sentence -> 1, chunk -> 0], []]        |
+--------------------------------------------------------------------------------------------+

{%- endcapture -%}

{%- capture approach_scala_example -%}
// In this example, the `rules.txt` has the form of
//
// the\s\w+, followed by 'the'
// ceremonies, ceremony
//
// where each regex is separated by the identifier by `","`
import ResourceHelper.spark.implicits._
import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.annotator.SentenceDetector
import com.johnsnowlabs.nlp.annotators.RegexMatcher
import org.apache.spark.ml.Pipeline

val documentAssembler = new DocumentAssembler().setInputCol("text").setOutputCol("document")

val sentence = new SentenceDetector().setInputCols("document").setOutputCol("sentence")

val regexMatcher = new RegexMatcher()
  .setExternalRules("src/test/resources/regex-matcher/rules.txt",  ",")
  .setInputCols(Array("sentence"))
  .setOutputCol("regex")
  .setStrategy("MATCH_ALL")

val pipeline = new Pipeline().setStages(Array(documentAssembler, sentence, regexMatcher))

val data = Seq(
  "My first sentence with the first rule. This is my second sentence with ceremonies rule."
).toDF("text")
val results = pipeline.fit(data).transform(data)

results.selectExpr("explode(regex) as result").show(false)
+--------------------------------------------------------------------------------------------+
|result                                                                                      |
+--------------------------------------------------------------------------------------------+
|[chunk, 23, 31, the first, [identifier -> followed by 'the', sentence -> 0, chunk -> 0], []]|
|[chunk, 71, 80, ceremonies, [identifier -> ceremony, sentence -> 1, chunk -> 0], []]        |
+--------------------------------------------------------------------------------------------+

{%- endcapture -%}

{%- capture approach_api_link -%}
[RegexMatcher](https://nlp.johnsnowlabs.com/api/com/johnsnowlabs/nlp/annotators/RegexMatcher)
{%- endcapture -%}

{%- capture approach_python_api_link -%}
[RegexMatcher](/api/python/reference/autosummary/python/sparknlp/annotator/matcher/regex_matcher/index.html#sparknlp.annotator.matcher.regex_matcher.RegexMatcher)
{%- endcapture -%}

{%- capture approach_source_link -%}
[RegexMatcher](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/RegexMatcher.scala)
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
