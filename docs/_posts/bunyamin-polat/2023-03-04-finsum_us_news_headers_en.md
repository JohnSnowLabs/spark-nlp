---
layout: model
title: US Financial News Articles Summarization(Headers)
author: John Snow Labs
name: finsum_us_news_headers
date: 2023-03-04
tags: [en, licensed, finance, summarization, t5, tensorflow]
task: Summarization
language: en
edition: Finance NLP 1.0.0
spark_version: 3.0
supported: true
engine: tensorflow
annotator: T5Transformer
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model is fine-tuned with US news articles from `bloomberg.com`, `cnbc.com`, `reuters.com`, `wsj.com`, and `fortune.com`. It is a Financial News Summarizer, aimed to extract headers from US financial news.

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/finance/models/finsum_us_news_headers_en_1.0.0_3.0_1677962901205.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/finance/models/finsum_us_news_headers_en_1.0.0_3.0_1677962901205.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
document_assembler = nlp.DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("documents")

t5 = nlp.T5Transformer().pretrained("finsum_us_news_headers", "en", "finance/models") \
    .setTask("summarization") \
    .setInputCols(["documents"]) \
    .setMaxOutputLength(512) \
    .setOutputCol("summaries")
    
data = spark.createDataFrame([["Kindred Systems Inc. was founded with the mission of creating human-like intelligence in machines and a vision to commercialize its research work in tandem. Over the past few years, Kindred’s Product and Artificial General Intelligence (AGI) divisions have accomplished a tremendous amount in their respective domains, working independently to allow each team to optimize for their objectives. The company has reached a point in its evolution where spinning off the AGI division maximizes the likelihood of success for both divisions, as well as returns to Kindred shareholders. Geordie Rose is stepping down as CEO and President of Kindred to lead this new entity named Sanctuary based in Vancouver, Canada. Kindred co-founder Suzanne Gildert will also be stepping down from her role as Chief Science Officer and will join Sanctuary as co-CEO. Sanctuary’s focus is on the implementation and testing of a specific framework for artificial general intelligence. The new entity will license some of Kindred’s patents and software, and Kindred will maintain a minority ownership in Sanctuary. Kindred’s Board of Directors has appointed Jim Liefer, previously the company’s COO, to serve as CEO and President. As COO, Liefer brought strong executive leadership alongside co-founder, George Babu, for the development and deployment of Kindred’s first commercial product Sort, and will continue to lead the company in its mission to research and develop human-like intelligence in machines. The Kindred team in Toronto will continue its applied research in machine and reinforcement learning, with the San Francisco office focused on robotics, product development, and commercialization. With Kindred Sort, the company aims to alleviate the massive pressures facing the retail and fulfillment industry, which includes significant online sales growth, labor shortages, and a lack of advancement in technology. Kindred Sort allows retailers to manage the exploding growth and demand of this sector more efficiently. During the 2017 holiday season, Kindred’s robots sorted thousands of items ordered at speeds averaging over 410 units per hour, and reaching peak speeds of over 531 units per hour, freeing human workers to perform other parts of the fulfillment process critical to meet growing customer demand. “Kindred will maintain its commitment to building human-like intelligence in machines and applying those learnings to create and teach a new intelligent class of robots that will enhance the quality of our day-to-day lives, and in particular, the way we work,” said Liefer. “We look forward to advancing Kindred Sort, achieving new AI and robotic milestones while also helping to drive retail and other industries forward.” Babu, Kindred co-founder, and Chief Product Officer will be joining Liefer on Kindred’s Board of Directors. Babu will continue to oversee product strategy and the expansion of Kindred’s partnerships and pilot programs with major global retailers. Kindred Systems Inc.’s mission is to build machines with human-like intelligence. The company’s central thesis is that intelligence requires a body. Since its founding in 2014, Kindred has been exploring and engineering systems that enable robots to understand and participate in our world, with the ultimate goal of a future where intelligent machines work together with people. Kindred is headquartered in San Francisco with an office in Toronto."]]).toDF("text")

pipeline = nlp.Pipeline().setStages([document_assembler, t5])

results = pipeline.fit(data).transform(data)
```

</div>

## Results

```bash
Kindred Systems Inc. Announces Appointment of Jim Liefer as CEO and President
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|finsum_us_news_headers|
|Compatibility:|Finance NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[documents]|
|Output Labels:|[summaries]|
|Language:|en|
|Size:|925.8 MB|

## References

Train dataset available [here](https://www.kaggle.com/datasets/jeet2016/us-financial-news-articles)
