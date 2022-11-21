{%- capture title -%}
BigTextMatcher
{%- endcapture -%}

{%- capture model_description -%}
Instantiated model of the BigTextMatcher.
For usage and examples see the documentation of the main class.
{%- endcapture -%}

{%- capture model_input_anno -%}
DOCUMENT, TOKEN
{%- endcapture -%}

{%- capture model_output_anno -%}
CHUNK
{%- endcapture -%}

{%- capture model_api_link -%}
[BigTextMatcherModel](https://nlp.johnsnowlabs.com/api/com/johnsnowlabs/nlp/annotators/btm/BigTextMatcherModel)
{%- endcapture -%}

{%- capture model_python_api_link -%}
[BigTextMatcherModel](/api/python/reference/autosummary/python/sparknlp/annotator/matcher/big_text_matcher/index.html#sparknlp.annotator.matcher.big_text_matcher.BigTextMatcherModel)
{%- endcapture -%}

{%- capture model_source_link -%}
[BigTextMatcherModel](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/btm/BigTextMatcherModel.scala)
{%- endcapture -%}

{%- capture approach_description -%}
Annotator to match exact phrases (by token) provided in a file against a Document.

A text file of predefined phrases must be provided with `setStoragePath`.


In contrast to the normal `TextMatcher`, the `BigTextMatcher` is designed for large corpora.

For extended examples of usage, see the [BigTextMatcherTestSpec](https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/annotators/btm/BigTextMatcherTestSpec.scala).
{%- endcapture -%}

{%- capture approach_input_anno -%}
DOCUMENT, TOKEN
{%- endcapture -%}

{%- capture approach_output_anno -%}
CHUNK
{%- endcapture -%}

{%- capture approach_python_example -%}
import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline
# In this example, the entities file is of the form
#
# ...
# dolore magna aliqua
# lorem ipsum dolor. sit
# laborum
# ...
#
# where each line represents an entity phrase to be extracted.

documentAssembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")

tokenizer = Tokenizer() \
    .setInputCols("document") \
    .setOutputCol("token")

data = spark.createDataFrame([["Hello dolore magna aliqua. Lorem ipsum dolor. sit in laborum"]]).toDF("text")
entityExtractor = BigTextMatcher() \
    .setInputCols("document", "token") \
    .setStoragePath("src/test/resources/entity-extractor/test-phrases.txt", ReadAs.TEXT) \
    .setOutputCol("entity") \
    .setCaseSensitive(False)

pipeline = Pipeline().setStages([documentAssembler, tokenizer, entityExtractor])
results = pipeline.fit(data).transform(data)
results.selectExpr("explode(entity)").show(truncate=False)
+--------------------------------------------------------------------+
|col                                                                 |
+--------------------------------------------------------------------+
|[chunk, 6, 24, dolore magna aliqua, [sentence -> 0, chunk -> 0], []]|
|[chunk, 53, 59, laborum, [sentence -> 0, chunk -> 1], []]           |
+--------------------------------------------------------------------+

{%- endcapture -%}

{%- capture approach_scala_example -%}
// In this example, the entities file is of the form
//
// ...
// dolore magna aliqua
// lorem ipsum dolor. sit
// laborum
// ...
//
// where each line represents an entity phrase to be extracted.
import spark.implicits._
import com.johnsnowlabs.nlp.DocumentAssembler
import com.johnsnowlabs.nlp.annotator.Tokenizer
import com.johnsnowlabs.nlp.annotator.BigTextMatcher
import com.johnsnowlabs.nlp.util.io.ReadAs
import org.apache.spark.ml.Pipeline

val documentAssembler = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val tokenizer = new Tokenizer()
  .setInputCols("document")
  .setOutputCol("token")

val data = Seq("Hello dolore magna aliqua. Lorem ipsum dolor. sit in laborum").toDF("text")
val entityExtractor = new BigTextMatcher()
  .setInputCols("document", "token")
  .setStoragePath("src/test/resources/entity-extractor/test-phrases.txt", ReadAs.TEXT)
  .setOutputCol("entity")
  .setCaseSensitive(false)

val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, entityExtractor))
val results = pipeline.fit(data).transform(data)
results.selectExpr("explode(entity)").show(false)
+--------------------------------------------------------------------+
|col                                                                 |
+--------------------------------------------------------------------+
|[chunk, 6, 24, dolore magna aliqua, [sentence -> 0, chunk -> 0], []]|
|[chunk, 53, 59, laborum, [sentence -> 0, chunk -> 1], []]           |
+--------------------------------------------------------------------+

{%- endcapture -%}

{%- capture approach_api_link -%}
[BigTextMatcher](https://nlp.johnsnowlabs.com/api/com/johnsnowlabs/nlp/annotators/btm/BigTextMatcher)
{%- endcapture -%}

{%- capture approach_python_api_link -%}
[BigTextMatcher](/api/python/reference/autosummary/python/sparknlp/annotator/matcher/big_text_matcher/index.html#sparknlp.annotator.matcher.big_text_matcher.BigTextMatcher)
{%- endcapture -%}

{%- capture approach_source_link -%}
[BigTextMatcher](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/btm/BigTextMatcher.scala)
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
