{%- capture title -%} Phi2Transformer {%- endcapture -%} 
{%- capture description -%} 
Phi-2 Transformer is a small, high-performance, open-source causal language model developed by Microsoft. With 2.7 billion parameters, it is trained on a mixture of synthetic and web datasets designed to improve commonsense reasoning and language understanding. Phi-2 demonstrates strong performance across reasoning, language comprehension, and benchmark evaluations, often outperforming models many times larger.  

It is particularly efficient for research, prototyping, and production tasks where small but powerful models are preferred.  

Pretrained models can be loaded with the `pretrained` method of the companion object:
```scala
val phi2 = Phi2Transformer.pretrained()
     .setInputCols("document")
     .setOutputCol("generation")
```

For available pretrained models please see the [Models Hub](https://sparknlp.org/models?annotator=Phi2Transformer).  

Spark NLP also supports a variety of Hugging Face transformer-based language models. Learn more here:  
- [Import models into Spark NLP](https://github.com/JohnSnowLabs/spark-nlp/discussions/5669)

**Resources**:
- [Phi-2 on Hugging Face](https://huggingface.co/microsoft/phi-2)  
- [Microsoft Research Blog on Phi-2](https://www.microsoft.com/en-us/research/blog/phi-2-the-surprising-power-of-small-language-models/)   
- [Arxiv: Textbooks Are All You Need II: phi-1.5 technical report](https://arxiv.org/abs/2309.05463)  

**Paper abstract**

*Over the past few months, our Machine Learning Foundations team at Microsoft Research has released a suite of small language models (SLMs) called “Phi” that achieve remarkable performance on a variety of benchmarks. Our first model, the 1.3 billion parameter Phi-1 (opens in new tab), achieved state-of-the-art performance on Python coding among existing SLMs (specifically on the HumanEval and MBPP benchmarks). We then extended our focus to common sense reasoning and language understanding and created a new 1.3 billion parameter model named Phi-1.5 (opens in new tab), with performance comparable to models 5x larger. We are now releasing Phi-2 (opens in new tab), a 2.7 billion-parameter language model that demonstrates outstanding reasoning and language understanding capabilities, showcasing state-of-the-art performance among base language models with less than 13 billion parameters. On complex benchmarks Phi-2 matches or outperforms models up to 25x larger, thanks to new innovations in model scaling and training data curation. With its compact size, Phi-2 is an ideal playground for researchers, including for exploration around mechanistic interpretability, safety improvements, or fine-tuning experimentation on a variety of tasks. We have made Phi-2 (opens in new tab) available in the Azure AI Studio model catalog to foster research and development on language models.*
{%- endcapture -%}

{%- capture input_anno -%}
DOCUMENT
{%- endcapture -%}

{%- capture output_anno -%}
GENERATION
{%- endcapture -%}

{%- capture api_link -%}
[Phi2Transformer](/api/com/johnsnowlabs/nlp/annotators/seq2seq/Phi2Transformer.html)
{%- endcapture -%}

{%- capture python_api_link -%}
[Phi2Transformer](/api/python/reference/autosummary/sparknlp/annotator/seq2seq/phi2_transformer/index.html)
{%- endcapture -%}

{%- capture source_link -%}
[Phi2Transformer](https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/main/scala/com/johnsnowlabs/nlp/annotators/seq2seq/Phi2Transformer.scala)
{%- endcapture -%}

{%- capture python_example -%}
from sparknlp.base import DocumentAssembler
from sparknlp.annotator import Phi2Transformer
from pyspark.ml import Pipeline

documentAssembler = DocumentAssembler() \
      .setInputCol("text") \
      .setOutputCol("document")

phi2 = Phi2Transformer.pretrained() \
      .setMaxOutputLength(50) \
      .setDoSample(False) \
      .setInputCols(["document"]) \
      .setOutputCol("generation")

pipeline = Pipeline().setStages([
    documentAssembler,
    phi2
])

data = spark.createDataFrame([
    ["What is the capital of France?"]
]).toDF("text")

model = pipeline.fit(data)
result = model.transform(data)

result.select("generation.result").show()

+------------+
|      result|
+------------+
|     [Paris]|
+------------+
{%- endcapture -%}

{%- capture scala_example -%}
import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.annotators.seq2seq.Phi2Transformer
import org.apache.spark.ml.Pipeline

val documentAssembler = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val phi2 = Phi2Transformer.pretrained()
  .setInputCols("document")
  .setOutputCol("generation")

val pipeline = new Pipeline().setStages(Array(
  documentAssembler,
  phi2
))

val data = Seq("What is the capital of France?").toDF("text")

val model = pipeline.fit(data)
val result = model.transform(data)

result.select("generation.result").show()

+------------+
|      result|
+------------+
|     [Paris]|
+------------+
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
