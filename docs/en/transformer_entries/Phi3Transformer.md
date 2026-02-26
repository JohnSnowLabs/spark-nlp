{%- capture title -%} Phi3Transformer {%- endcapture -%} 
{%- capture description -%} 
Phi-3 is a family of small language models (SLMs) optimized for efficient reasoning, code generation, and general NLP tasks. The models range from 3.8B to 14B parameters, with context lengths up to 128K tokens, and are best suited for instruction-following and chat-style prompts. Trained on curated public, educational, and synthetic data, Phi-3 emphasizes strong performance in math, coding, and logical reasoning while remaining lightweight and cost-effective. The design enables fast, affordable deployment across cloud and edge environments, with a primary focus on English (limited multilingual support).

Pretrained models can be loaded with the `pretrained` method of the companion object:
```scala
val phi3 = Phi3Transformer.pretrained()
     .setInputCols("document")
     .setOutputCol("generation")
```
The default model is `"phi_3.5_mini_instruct_int4"`, if no name is provided.

For available pretrained models please see the [Models Hub](https://sparknlp.org/models?annotator=Phi3Transformer).

Spark NLP also supports Hugging Face transformer-based translation models. Learn more here:  
- [Import models into Spark NLP](https://github.com/JohnSnowLabs/spark-nlp/discussions/5669)

**Resources**:
- [Introducing Phi-3: Redefining What’s Possible with SLMs (Microsoft Blog)](https://azure.microsoft.com/en-us/blog/introducing-phi-3-redefining-whats-possible-with-slms/)  
- [Phi-3 Small Language Models with Big Potential (Microsoft Source)](https://news.microsoft.com/source/features/ai/the-phi-3-small-language-models-with-big-potential/)  
- [Phi-3 Tutorial and Overview (DataCamp)](https://www.datacamp.com/tutorial/phi-3-tutorial)  
- [Evaluation of Phi-3 Models (Arxiv)](https://arxiv.org/abs/2404.14219)  

**Paper abstract**

*We introduce phi-3-mini, a 3.8 billion parameter language model trained on 3.3 trillion tokens, whose overall performance, as measured by both academic benchmarks and internal testing, rivals that of models such as Mixtral 8x7B and GPT-3.5 (e.g., phi-3-mini achieves 69% on MMLU and 8.38 on MT-bench), despite being small enough to be deployed on a phone. The innovation lies entirely in our dataset for training, a scaled-up version of the one used for phi-2, composed of heavily filtered web data and synthetic data. The model is also further aligned for robustness, safety, and chat format. We also provide some initial parameter-scaling results with a 7B and 14B models trained for 4.8T tokens, called phi-3-small and phi-3-medium, both significantly more capable than phi-3-mini (e.g., respectively 75% and 78% on MMLU, and 8.7 and 8.9 on MT-bench).*
{%- endcapture -%}

{%- capture input_anno -%}
DOCUMENT
{%- endcapture -%}

{%- capture output_anno -%}
GENERATION
{%- endcapture -%}

{%- capture api_link -%}
[Phi3Transformer](/api/com/johnsnowlabs/nlp/annotators/seq2seq/Phi3Transformer.html)
{%- endcapture -%}

{%- capture python_api_link -%}
[Phi3Transformer](/api/python/reference/autosummary/sparknlp/annotator/seq2seq/phi3_transformer/index.html)
{%- endcapture -%}

{%- capture source_link -%}
[Phi3Transformer](https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/main/scala/com/johnsnowlabs/nlp/annotators/seq2seq/Phi3Transformer.scala)
{%- endcapture -%}

{%- capture python_example -%}
from sparknlp.base import DocumentAssembler
from sparknlp.annotator import Phi3Transformer
from pyspark.ml import Pipeline

documentAssembler = DocumentAssembler() \
      .setInputCol("text") \
      .setOutputCol("document")   

phi3 = Phi3Transformer.pretrained() \
      .setInputCols(["document"]) \
      .setOutputCol("generation") 

pipeline = Pipeline().setStages([
    documentAssembler, 
    embephi3ddings
])

data = spark.createDataFrame([
    ["What is 7 × 8?"]
]).toDF("text")

model = pipeline.fit(data)
result = model.transform(data)

result.select("generation.result").show()

+-------------------------------+
|result                         |
+-------------------------------+
|[The product of 7 and 8 is 56.]|
+-------------------------------+
{%- endcapture -%}

{%- capture scala_example -%}
import com.johnsnowlabs.nlp.base._
import com.johnsnowlabs.nlp.annotator._
import org.apache.spark.ml.Pipeline

val documentAssembler = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val phi3 = Phi3Transformer.pretrained()
  .setInputCols("document")
  .setOutputCol("generation")

val pipeline = new Pipeline().setStages(Array(
  documentAssembler,
  phi3
))

val data = Seq("What is 7 × 8?").toDF("text")

val model = pipeline.fit(data)
val result = model.transform(data)

result.select("generation.result").show()

+-------------------------------+
|result                         |
+-------------------------------+
|[The product of 7 and 8 is 56.]|
+-------------------------------+
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