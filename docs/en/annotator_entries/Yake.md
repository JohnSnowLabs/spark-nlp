{%- capture title -%}
YakeKeywordExtraction
{%- endcapture -%}

{%- capture description -%}
Yake is an Unsupervised, Corpus-Independent, Domain and Language-Independent and Single-Document keyword extraction
algorithm.

Extracting keywords from texts has become a challenge for individuals and organizations as the information grows in
complexity and size. The need to automate this task so that text can be processed in a timely and adequate manner has
led to the emergence of automatic keyword extraction tools. Yake is a novel feature-based system for multi-lingual
keyword extraction, which supports texts of different sizes, domain or languages. Unlike other approaches, Yake does
not rely on dictionaries nor thesauri, neither is trained against any corpora. Instead, it follows an unsupervised
approach which builds upon features extracted from the text, making it thus applicable to documents written in
different languages without the need for further knowledge. This can be beneficial for a large number of tasks and a
plethora of situations where access to training corpora is either limited or restricted.
The algorithm makes use of the position of a sentence and token. Therefore, to use the annotator, the text should be
first sent through a Sentence Boundary Detector and then a tokenizer.

Note that each keyword will be given a keyword score greater than 0 (The lower the score better the keyword).
Therefore to filter the keywords, an upper bound for the score can be set with `setThreshold`.

For extended examples of usage, see the [Spark NLP Workshop](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Public/8.Keyword_Extraction_YAKE.ipynb)
and the [YakeTestSpec](https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/annotators/keyword/yake/YakeTestSpec.scala).

 **Sources** :

[Campos, R., Mangaravite, V., Pasquali, A., Jatowt, A., Jorge, A., Nunes, C. and Jatowt, A. (2020). YAKE! Keyword Extraction from Single Documents using Multiple Local Features. In Information Sciences Journal. Elsevier, Vol 509, pp 257-289](https://www.sciencedirect.com/science/article/pii/S0020025519308588)

**Paper abstract:**

*As the amount of generated information grows, reading and summarizing texts of large collections turns into a challenging task. Many documents do not come with descriptive terms,
thus requiring humans to generate keywords on-the-fly. The need to automate this kind of task demands the development of keyword extraction systems with the ability to automatically
identify keywords within the text. One approach is to resort to machine-learning algorithms. These, however, depend on large annotated text corpora, which are not always available.
An alternative solution is to consider an unsupervised approach. In this article, we describe YAKE!, a light-weight unsupervised automatic keyword extraction method which rests on
statistical text features extracted from single documents to select the most relevant keywords of a text. Our system does not need to be trained on a particular set of documents,
nor does it depend on dictionaries, external corpora, text size, language, or domain. To demonstrate the merits and significance of YAKE!, we compare it against ten state-of-the-art
unsupervised approaches and one supervised method. Experimental results carried out on top of twenty datasets show that YAKE! significantly outperforms other unsupervised methods on
texts of different sizes, languages, and domains.*
{%- endcapture -%}

{%- capture input_anno -%}
TOKEN
{%- endcapture -%}

{%- capture output_anno -%}
CHUNK
{%- endcapture -%}

{%- capture python_example -%}
import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline

documentAssembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")

sentenceDetector = SentenceDetector() \
    .setInputCols(["document"]) \
    .setOutputCol("sentence")

token = Tokenizer() \
    .setInputCols(["sentence"]) \
    .setOutputCol("token") \
    .setContextChars(["(", "]", "?", "!", ".", ","])

keywords = YakeKeywordExtraction() \
    .setInputCols(["token"]) \
    .setOutputCol("keywords") \
    .setThreshold(0.6) \
    .setMinNGrams(2) \
    .setNKeywords(10)

pipeline = Pipeline().setStages([
    documentAssembler,
    sentenceDetector,
    token,
    keywords
])

data = spark.createDataFrame([[
    "Sources tell us that Google is acquiring Kaggle, a platform that hosts data science and machine learning competitions. Details about the transaction remain somewhat vague, but given that Google is hosting its Cloud Next conference in San Francisco this week, the official announcement could come as early as tomorrow. Reached by phone, Kaggle co-founder CEO Anthony Goldbloom declined to deny that the acquisition is happening. Google itself declined 'to comment on rumors'. Kaggle, which has about half a million data scientists on its platform, was founded by Goldbloom  and Ben Hamner in 2010. The service got an early start and even though it has a few competitors like DrivenData, TopCoder and HackerRank, it has managed to stay well ahead of them by focusing on its specific niche. The service is basically the de facto home for running data science and machine learning competitions. With Kaggle, Google is buying one of the largest and most active communities for data scientists - and with that, it will get increased mindshare in this community, too (though it already has plenty of that thanks to Tensorflow and other projects). Kaggle has a bit of a history with Google, too, but that's pretty recent. Earlier this month, Google and Kaggle teamed up to host a $100,000 machine learning competition around classifying YouTube videos. That competition had some deep integrations with the Google Cloud Platform, too. Our understanding is that Google will keep the service running - likely under its current name. While the acquisition is probably more about Kaggle's community than technology, Kaggle did build some interesting tools for hosting its competition and 'kernels', too. On Kaggle, kernels are basically the source code for analyzing data sets and developers can share this code on the platform (the company previously called them 'scripts'). Like similar competition-centric sites, Kaggle also runs a job board, too. It's unclear what Google will do with that part of the service. According to Crunchbase, Kaggle raised $12.5 million (though PitchBook says it's $12.75) since its   launch in 2010. Investors in Kaggle include Index Ventures, SV Angel, Max Levchin, NaRavikant, Google chie economist Hal Varian, Khosla Ventures and Yuri Milner"
]]).toDF("text")
result = pipeline.fit(data).transform(data)

# combine the result and score (contained in keywords.metadata)
scores = result \
    .selectExpr("explode(arrays_zip(keywords.result, keywords.metadata)) as resultTuples") \
    .selectExpr("resultTuples['0'] as keyword", "resultTuples['1'].score as score")

# Order ascending, as lower scores means higher importance
scores.orderBy("score").show(5, truncate = False)
+---------------------+-------------------+
|keyword              |score              |
+---------------------+-------------------+
|google cloud         |0.32051516486864573|
|google cloud platform|0.37786450577630676|
|ceo anthony goldbloom|0.39922830978423146|
|san francisco        |0.40224744669493756|
|anthony goldbloom    |0.41584827825302534|
+---------------------+-------------------+

{%- endcapture -%}

{%- capture scala_example -%}
import spark.implicits._
import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.annotator.{SentenceDetector, Tokenizer}
import com.johnsnowlabs.nlp.annotators.keyword.yake.YakeKeywordExtraction
import org.apache.spark.ml.Pipeline

val documentAssembler = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val sentenceDetector = new SentenceDetector()
  .setInputCols("document")
  .setOutputCol("sentence")

val token = new Tokenizer()
  .setInputCols("sentence")
  .setOutputCol("token")
  .setContextChars(Array("(", ")", "?", "!", ".", ","))

val keywords = new YakeKeywordExtraction()
  .setInputCols("token")
  .setOutputCol("keywords")
  .setThreshold(0.6f)
  .setMinNGrams(2)
  .setNKeywords(10)

val pipeline = new Pipeline().setStages(Array(
  documentAssembler,
  sentenceDetector,
  token,
  keywords
))

val data = Seq(
  "Sources tell us that Google is acquiring Kaggle, a platform that hosts data science and machine learning competitions. Details about the transaction remain somewhat vague, but given that Google is hosting its Cloud Next conference in San Francisco this week, the official announcement could come as early as tomorrow. Reached by phone, Kaggle co-founder CEO Anthony Goldbloom declined to deny that the acquisition is happening. Google itself declined 'to comment on rumors'. Kaggle, which has about half a million data scientists on its platform, was founded by Goldbloom  and Ben Hamner in 2010. The service got an early start and even though it has a few competitors like DrivenData, TopCoder and HackerRank, it has managed to stay well ahead of them by focusing on its specific niche. The service is basically the de facto home for running data science and machine learning competitions. With Kaggle, Google is buying one of the largest and most active communities for data scientists - and with that, it will get increased mindshare in this community, too (though it already has plenty of that thanks to Tensorflow and other projects). Kaggle has a bit of a history with Google, too, but that's pretty recent. Earlier this month, Google and Kaggle teamed up to host a $100,000 machine learning competition around classifying YouTube videos. That competition had some deep integrations with the Google Cloud Platform, too. Our understanding is that Google will keep the service running - likely under its current name. While the acquisition is probably more about Kaggle's community than technology, Kaggle did build some interesting tools for hosting its competition and 'kernels', too. On Kaggle, kernels are basically the source code for analyzing data sets and developers can share this code on the platform (the company previously called them 'scripts'). Like similar competition-centric sites, Kaggle also runs a job board, too. It's unclear what Google will do with that part of the service. According to Crunchbase, Kaggle raised $12.5 million (though PitchBook says it's $12.75) since its   launch in 2010. Investors in Kaggle include Index Ventures, SV Angel, Max Levchin, Naval Ravikant, Google chief economist Hal Varian, Khosla Ventures and Yuri Milner"
).toDF("text")
val result = pipeline.fit(data).transform(data)

// combine the result and score (contained in keywords.metadata)
val scores = result
  .selectExpr("explode(arrays_zip(keywords.result, keywords.metadata)) as resultTuples")
  .select($"resultTuples.0" as "keyword", $"resultTuples.1.score")

// Order ascending, as lower scores means higher importance
scores.orderBy("score").show(5, truncate = false)
+---------------------+-------------------+
|keyword              |score              |
+---------------------+-------------------+
|google cloud         |0.32051516486864573|
|google cloud platform|0.37786450577630676|
|ceo anthony goldbloom|0.39922830978423146|
|san francisco        |0.40224744669493756|
|anthony goldbloom    |0.41584827825302534|
+---------------------+-------------------+

{%- endcapture -%}

{%- capture api_link -%}
[YakeKeywordExtraction](https://nlp.johnsnowlabs.com/api/com/johnsnowlabs/nlp/annotators/keyword/yake/YakeKeywordExtraction)
{%- endcapture -%}

{%- capture python_api_link -%}
[YakeKeywordExtraction](/api/python/reference/autosummary/python/sparknlp/annotator/keyword_extraction/yake_keyword_extraction/index.html#sparknlp.annotator.keyword_extraction.yake_keyword_extraction.YakeKeywordExtraction)
{%- endcapture -%}

{%- capture source_link -%}
[YakeKeywordExtraction](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/keyword.yake/YakeKeywordExtraction.scala)
{%- endcapture -%}

{% include templates/anno_template.md
title=title
description=description
input_anno=input_anno
output_anno=output_anno
python_example=python_example
scala_example=scala_example
python_api_link=python_api_link
api_link=api_link
source_link=source_link
%}