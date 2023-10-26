{%- capture title -%}
OpenAICompletion
{%- endcapture -%}

{%- capture description -%}
Transformer that makes a request for OpenAI Completion API for each executor.

See the [OpenAI API Doc](https://platform.openai.com/docs/api-reference/completions/create) for reference.
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
from sparknlp.common import *
from pyspark.ml import Pipeline

documentAssembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")
openai_completion = OpenAICompletion() \
    .setInputCols("document") \
    .setOutputCol("completion") \
    .setModel("text-davinci-003") \
    .setMaxTokens(100)
pipeline = Pipeline().setStages([
    documentAssembler,
    openai_completion
])

empty_df = spark.createDataFrame([[""]], ["text"])
sample_text= [["Generate a restaurant review."], ["Write a review for a local eatery."], ["Create a JSON with a review of a dining experience."]]
sample_df = spark.createDataFrame(sample_text).toDF("text")
sample_df.show()
+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|completion                                                                                                                                                                                                                                                                                        |
+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|[{document, 0, 258, \n\nI had the pleasure of dining at La Fiorita recently, and it was a truly delightful experience! The menu boasted a wonderful selection of classic Italian dishes, all exquisitely prepared and presented. The service staff was friendly and attentive and really, {}, []}]|
|[{document, 0, 227, \n\nI recently visited Barbecue Joe's for dinner and it was amazing! The menu had so many items to choose from including pulled pork, smoked turkey, brisket, pork ribs, and sandwiches. I opted for the pulled pork sandwich and let, {}, []}]                               |
|[{document, 0, 172, \n\n{ \n   "review": { \n      "overallRating": 4, \n      "reviewBody": "I enjoyed my meal at this restaurant. The food was flavourful, well-prepared and beautifully presented., {}, []}]                                                                                   |
+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
{%- endcapture -%}

{%- capture scala_example -%}
import spark.implicits._
import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.ml.ai.OpenAICompletion
import org.apache.spark.ml.Pipeline

val documentAssembler = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val openAICompletion = new OpenAICompletion()
 .setInputCols("document")
 .setOutputCol("completion")
 .setModel("text-davinci-003")
 .setMaxTokens(50)


val pipeline = new Pipeline().setStages(Array(
  documentAssembler,
  openAICompletion
))

val promptDF = Seq(
 "Generate a restaurant review.",
 "Write a review for a local eatery.",
 "Create a JSON with a review of a dining experience.").toDS.toDF("text")
val completionDF = pipeline.fit(promptDF).transform(promptDF)

completionDF.select("completion").show(false)
+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|completion                                                                                                                                                                                                                                                                                        |
+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|[{document, 0, 258, \n\nI had the pleasure of dining at La Fiorita recently, and it was a truly delightful experience! The menu boasted a wonderful selection of classic Italian dishes, all exquisitely prepared and presented. The service staff was friendly and attentive and really, {}, []}]|
|[{document, 0, 227, \n\nI recently visited Barbecue Joe's for dinner and it was amazing! The menu had so many items to choose from including pulled pork, smoked turkey, brisket, pork ribs, and sandwiches. I opted for the pulled pork sandwich and let, {}, []}]                               |
|[{document, 0, 172, \n\n{ \n   "review": { \n      "overallRating": 4, \n      "reviewBody": "I enjoyed my meal at this restaurant. The food was flavourful, well-prepared and beautifully presented., {}, []}]                                                                                   |
+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+

{%- endcapture -%}

{%- capture api_link -%}
[OpenAICompletion](/api/com/johnsnowlabs/ml/ai/OpenAICompletion)
{%- endcapture -%}

{%- capture python_api_link -%}
[OpenAICompletion](/api/python/reference/autosummary/sparknlp/annotator/openai/openai_completion/index.html#sparknlp.annotator.openai.openai_completion.OpenAICompletion)
{%- endcapture -%}

{%- capture source_link -%}
[OpenAICompletion](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/ml/ai/OpenAICompletion.scala)
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