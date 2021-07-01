{%- capture title -%}
BertEmbeddings
{%- endcapture -%}

{%- capture description -%}
Token-level embeddings using BERT. BERT (Bidirectional Encoder Representations from Transformers) provides dense
vector representations for natural language by using a deep, pre-trained neural network with the Transformer architecture.

Pretrained models can be loaded with `pretrained` of the companion object:
```
val embeddings = BertEmbeddings.pretrained()
  .setInputCols("token", "document")
  .setOutputCol("bert_embeddings")
```
The default model is `"small_bert_L2_768"`, if no name is provided.

For available pretrained models please see the [Models Hub](https://nlp.johnsnowlabs.com/models?task=Embeddings).

For extended examples of usage, see the [Spark NLP Workshop](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/blogposts/3.NER_with_BERT.ipynb)
and the [BertEmbeddingsTestSpec](https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/embeddings/BertEmbeddingsTestSpec.scala).

**Sources** :

[BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)

https://github.com/google-research/bert

**Paper abstract**

*We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations
from Transformers. Unlike recent language representation models, BERT is designed to pre-train deep bidirectional
representations from unlabeled text by jointly conditioning on both left and right context in all layers. As a
result, the pre-trained BERT model can be fine-tuned with just one additional output layer to create
state-of-the-art models for a wide range of tasks, such as question answering and language inference, without
substantial task-specific architecture modifications. BERT is conceptually simple and empirically powerful. It
obtains new state-of-the-art results on eleven natural language processing tasks, including pushing the GLUE score
to 80.5% (7.7% point absolute improvement), MultiNLI accuracy to 86.7% (4.6% absolute improvement), SQuAD v1.1
question answering Test F1 to 93.2 (1.5 point absolute improvement) and SQuAD v2.0 Test F1 to 83.1 (5.1 point
absolute improvement).*
{%- endcapture -%}

{%- capture input_anno -%}
DOCUMENT, TOKEN
{%- endcapture -%}

{%- capture output_anno -%}
WORD_EMBEDDINGS
{%- endcapture -%}

{%- capture python_example -%}
import sparknlp
from sparknlp.base import *
from sparknlp.common import *
from sparknlp.annotator import *
from sparknlp.training import *
from pyspark.ml import Pipeline

documentAssembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")

tokenizer = Tokenizer() \
    .setInputCols(["document"]) \
    .setOutputCol("token")

embeddings = BertEmbeddings.pretrained("small_bert_L2_128", "en") \
    .setInputCols(["token", "document"]) \
    .setOutputCol("bert_embeddings")

embeddingsFinisher = EmbeddingsFinisher() \
    .setInputCols(["bert_embeddings"]) \
    .setOutputCols("finished_embeddings") \
    .setOutputAsVector(True)

pipeline = Pipeline().setStages([
    documentAssembler,
    tokenizer,
    embeddings,
    embeddingsFinisher
])

data = spark.createDataFrame([["This is a sentence."]]).toDF("text")
result = pipeline.fit(data).transform(data)

result.selectExpr("explode(finished_embeddings) as result").show(5, 80)
+--------------------------------------------------------------------------------+
|                                                                          result|
+--------------------------------------------------------------------------------+
|[-2.3497989177703857,0.480538547039032,-0.3238905668258667,-1.612930893898010...|
|[-2.1357314586639404,0.32984697818756104,-0.6032363176345825,-1.6791689395904...|
|[-1.8244884014129639,-0.27088963985443115,-1.059438943862915,-0.9817547798156...|
|[-1.1648050546646118,-0.4725411534309387,-0.5938255786895752,-1.5780693292617...|
|[-0.9125322699546814,0.4563939869403839,-0.3975459933280945,-1.81611204147338...|
+--------------------------------------------------------------------------------+

{%- endcapture -%}

{%- capture scala_example -%}
import spark.implicits._
import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.annotators.Tokenizer
import com.johnsnowlabs.nlp.embeddings.BertEmbeddings
import com.johnsnowlabs.nlp.EmbeddingsFinisher
import org.apache.spark.ml.Pipeline

val documentAssembler = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val tokenizer = new Tokenizer()
  .setInputCols("document")
  .setOutputCol("token")

val embeddings = BertEmbeddings.pretrained("small_bert_L2_128", "en")
  .setInputCols("token", "document")
  .setOutputCol("bert_embeddings")

val embeddingsFinisher = new EmbeddingsFinisher()
  .setInputCols("bert_embeddings")
  .setOutputCols("finished_embeddings")
  .setOutputAsVector(true)

val pipeline = new Pipeline().setStages(Array(
  documentAssembler,
  tokenizer,
  embeddings,
  embeddingsFinisher
))

val data = Seq("This is a sentence.").toDF("text")
val result = pipeline.fit(data).transform(data)

result.selectExpr("explode(finished_embeddings) as result").show(5, 80)
+--------------------------------------------------------------------------------+
|                                                                          result|
+--------------------------------------------------------------------------------+
|[-2.3497989177703857,0.480538547039032,-0.3238905668258667,-1.612930893898010...|
|[-2.1357314586639404,0.32984697818756104,-0.6032363176345825,-1.6791689395904...|
|[-1.8244884014129639,-0.27088963985443115,-1.059438943862915,-0.9817547798156...|
|[-1.1648050546646118,-0.4725411534309387,-0.5938255786895752,-1.5780693292617...|
|[-0.9125322699546814,0.4563939869403839,-0.3975459933280945,-1.81611204147338...|
+--------------------------------------------------------------------------------+

{%- endcapture -%}

{%- capture api_link -%}
[BertEmbeddings](https://nlp.johnsnowlabs.com/api/com/johnsnowlabs/nlp/embeddings/BertEmbeddings)
{%- endcapture -%}

{%- capture source_link -%}
[BertEmbeddings](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/embeddings/BertEmbeddings.scala)
{%- endcapture -%}

{% include templates/anno_template.md
title=title
description=description
input_anno=input_anno
output_anno=output_anno
python_example=python_example
scala_example=scala_example
api_link=api_link
source_link=source_link
%}