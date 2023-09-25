{%- capture title -%}
MPNetEmbeddings
{%- endcapture -%}

{%- capture description -%}
Sentence embeddings using MPNet.

The MPNet model was proposed in MPNet: Masked and Permuted Pre-training for Language
Understanding by Kaitao Song, Xu Tan, Tao Qin, Jianfeng Lu, Tie-Yan Liu. MPNet adopts a novel
pre-training method, named masked and permuted language modeling, to inherit the advantages of
masked language modeling and permuted language modeling for natural language understanding.

Pretrained models can be loaded with `pretrained` of the companion object:

```scala
val embeddings = MPNetEmbeddings.pretrained()
  .setInputCols("document")
  .setOutputCol("mpnet_embeddings")
```

The default model is `"all_mpnet_base_v2"`, if no name is provided.

For available pretrained models please see the
[Models Hub](https://sparknlp.org/models?q=MPNet).

For extended examples of usage, see
[MPNetEmbeddingsTestSpec](https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/embeddings/MPNetEmbeddingsTestSpec.scala).

**Sources** :

[MPNet: Masked and Permuted Pre-training for Language Understanding](https://arxiv.org/abs/2004.09297)

[MPNet Github Repository](https://github.com/microsoft/MPNet)

**Paper abstract**

*BERT adopts masked language modeling (MLM) for pre-training and is one of the most
successful pre-training models. Since BERT neglects dependency among predicted tokens, XLNet
introduces permuted language modeling (PLM) for pre-training to address this problem. However,
XLNet does not leverage the full position information of a sentence and thus suffers from
position discrepancy between pre-training and fine-tuning. In this paper, we propose MPNet, a
novel pre-training method that inherits the advantages of BERT and XLNet and avoids their
limitations. MPNet leverages the dependency among predicted tokens through permuted language
modeling (vs. MLM in BERT), and takes auxiliary position information as input to make the
model see a full sentence and thus reducing the position discrepancy (vs. PLM in XLNet). We
pre-train MPNet on a large-scale dataset (over 160GB text corpora) and fine-tune on a variety
of down-streaming tasks (GLUE, SQuAD, etc). Experimental results show that MPNet outperforms
MLM and PLM by a large margin, and achieves better results on these tasks compared with
previous state-of-the-art pre-trained methods (e.g., BERT, XLNet, RoBERTa) under the same
model setting.*
{%- endcapture -%}

{%- capture input_anno -%}
DOCUMENT
{%- endcapture -%}

{%- capture output_anno -%}
SENTENCE_EMBEDDINGS
{%- endcapture -%}

{%- capture python_example -%}
import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline

documentAssembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")
embeddings = MPNetEmbeddings.pretrained() \
    .setInputCols(["document"]) \
    .setOutputCol("mpnet_embeddings")
embeddingsFinisher = EmbeddingsFinisher() \
    .setInputCols(["mpnet_embeddings"]) \
    .setOutputCols("finished_embeddings") \
    .setOutputAsVector(True)
pipeline = Pipeline().setStages([
    documentAssembler,
    embeddings,
    embeddingsFinisher
])

data = spark.createDataFrame([["This is an example sentence", "Each sentence is converted"]]).toDF("text")
result = pipeline.fit(data).transform(data)
result.selectExpr("explode(finished_embeddings) as result").show(5, 80)
+--------------------------------------------------------------------------------+
|                                                                          result|
+--------------------------------------------------------------------------------+
|[[0.022502584, -0.078291744, -0.023030775, -0.0051000593, -0.080340415, 0.039...|
|[[0.041702367, 0.0010974605, -0.015534201, 0.07092203, -0.0017729357, 0.04661...|
+--------------------------------------------------------------------------------+
{%- endcapture -%}

{%- capture scala_example -%}
import spark.implicits._
import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.annotators.Tokenizer
import com.johnsnowlabs.nlp.embeddings.MPNetEmbeddings
import com.johnsnowlabs.nlp.EmbeddingsFinisher
import org.apache.spark.ml.Pipeline

val documentAssembler = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val embeddings = MPNetEmbeddings.pretrained("all_mpnet_base_v2", "en")
  .setInputCols("document")
  .setOutputCol("mpnet_embeddings")

val embeddingsFinisher = new EmbeddingsFinisher()
  .setInputCols("mpnet_embeddings")
  .setOutputCols("finished_embeddings")
  .setOutputAsVector(true)

val pipeline = new Pipeline().setStages(Array(
  documentAssembler,
  embeddings,
  embeddingsFinisher
))

val data = Seq("This is an example sentence", "Each sentence is converted").toDF("text")
val result = pipeline.fit(data).transform(data)

result.selectExpr("explode(finished_embeddings) as result").show(1, 80)
+--------------------------------------------------------------------------------+
|                                                                          result|
+--------------------------------------------------------------------------------+
|[[0.022502584, -0.078291744, -0.023030775, -0.0051000593, -0.080340415, 0.039...|
|[[0.041702367, 0.0010974605, -0.015534201, 0.07092203, -0.0017729357, 0.04661...|
+--------------------------------------------------------------------------------+

{%- endcapture -%}

{%- capture api_link -%}
[MPNetEmbeddings](/api/com/johnsnowlabs/nlp/embeddings/MPNetEmbeddings)
{%- endcapture -%}

{%- capture python_api_link -%}
[MPNetEmbeddings](/api/python/reference/autosummary/sparknlp/annotator/embeddings/mpnet_embeddings/index.html#sparknlp.annotator.embeddings.mpnet_embeddings.MPNetEmbeddings)
{%- endcapture -%}

{%- capture source_link -%}
[MPNetEmbeddings](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/embeddings/MPNetEmbeddings.scala)
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