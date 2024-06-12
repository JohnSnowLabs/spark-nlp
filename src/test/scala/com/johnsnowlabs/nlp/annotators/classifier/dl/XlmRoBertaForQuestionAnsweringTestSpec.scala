/*
 * Copyright 2017-2022 John Snow Labs
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.johnsnowlabs.nlp.annotators.classifier.dl

import com.johnsnowlabs.nlp.annotators.Tokenizer
import com.johnsnowlabs.nlp.base._
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.tags.SlowTest
import com.johnsnowlabs.util.Benchmark
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.scalatest.flatspec.AnyFlatSpec

class XlmRoBertaForQuestionAnsweringTestSpec extends AnyFlatSpec {

  import ResourceHelper.spark.implicits._

  "XlmRoBertaForQuestionAnswering" should "correctly load custom model with extracted signatures" taggedAs SlowTest in {

    val beyonceContext =
      """Beyoncé Giselle Knowles-Carter (/biːˈjɒnseɪ/ bee-YON-say) (born September 4, 1981) is an American singer, songwriter, record producer and actress. Born and raised in Houston, Texas, she performed in various singing and dancing competitions as a child, and rose to fame in the late 1990s as lead singer of R&B girl-group Destiny's Child. Managed by her father, Mathew Knowles, the group became one of the world's best-selling girl groups of all time. Their hiatus saw the release of Beyoncé's debut album, Dangerously in Love (2003), which established her as a solo artist worldwide, earned five Grammy Awards and featured the Billboard Hot 100 number-one singles "Crazy in Love" and "Baby Boy"."""
    val amazonContext =
      """The Amazon rainforest (Portuguese: Floresta Amazônica or Amazônia; Spanish: Selva Amazónica, Amazonía or usually Amazonia; French: Forêt amazonienne; Dutch: Amazoneregenwoud), also known in English as Amazonia or the Amazon Jungle, is a moist broadleaf forest that covers most of the Amazon basin of South America. This basin encompasses 7,000,000 square kilometres (2,700,000 sq mi), of which 5,500,000 square kilometres (2,100,000 sq mi) are covered by the rainforest. This region includes territory belonging to nine nations. The majority of the forest is contained within Brazil, with 60% of the rainforest, followed by Peru with 13%, Colombia with 10%, and with minor amounts in Venezuela, Ecuador, Bolivia, Guyana, Suriname and French Guiana. States or departments in four nations contain "Amazonas" in their names. The Amazon represents over half of the planet's remaining rainforests, and comprises the largest and most biodiverse tract of tropical rainforest in the world, with an estimated 390 billion individual trees divided into 16,000 species."""

    val ddd = Seq(
      (
        "Where was John Lenon born?",
        "John Lenon was born in London and lived in Paris. My name is Sarah and I live in London."),
      ("What's my name?", "My name is Clara and I live in Berkeley."),
      ("Which name is also used to describe the Amazon rainforest in English?", amazonContext),
      ("When did Beyonce start becoming popular?", beyonceContext),
      ("What areas did Beyonce compete in when she was growing up?", beyonceContext),
      ("When did Beyonce leave Destiny's Child and become a solo singer?", beyonceContext),
      ("What was the first album Beyoncé released as a solo artist?", beyonceContext))
      .toDF("question", "context")
      .repartition(1)

    val document = new MultiDocumentAssembler()
      .setInputCols("question", "context")
      .setOutputCols("document_question", "document_context")

    val questionAnswering = XlmRoBertaForQuestionAnswering
      .pretrained()
      .setInputCols(Array("document_question", "document_context"))
      .setOutputCol("answer")
      .setCaseSensitive(true)
      .setMaxSentenceLength(512)

    val pipeline = new Pipeline().setStages(Array(document, questionAnswering))

    val pipelineModel = pipeline.fit(ddd)
    val pipelineDF = pipelineModel.transform(ddd)

    pipelineDF.show(false)
    pipelineDF.select("answer").show(false)
    pipelineDF.select("answer.result").show(false)

  }

  "XlmRoBertaForQuestionAnswering" should "be saved and loaded correctly" taggedAs SlowTest in {

    import ResourceHelper.spark.implicits._

    val ddd = Seq(
      "John Lenon was born in London and lived in Paris. My name is Sarah and I live in London",
      "Rare Hendrix song draft sells for almost $17,000.",
      "EU rejects German call to boycott British lamb .",
      "TORONTO 1996-08-21").toDF("text")

    val document = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val tokenizer = new Tokenizer()
      .setInputCols(Array("document"))
      .setOutputCol("token")

    val questionAnswering = XlmRoBertaForQuestionAnswering
      .pretrained()
      .setInputCols(Array("token", "document"))
      .setOutputCol("label")
      .setCaseSensitive(true)

    val pipeline = new Pipeline().setStages(Array(document, tokenizer, questionAnswering))

    val pipelineModel = pipeline.fit(ddd)
    val pipelineDF = pipelineModel.transform(ddd)

    pipelineDF.select("label.result").show(false)

    Benchmark.time("Time to save XlmRoBertaForQuestionAnswering pipeline model") {
      pipelineModel.write.overwrite().save("./tmp_xlmrobertaforquestion_pipeline")
    }

    Benchmark.time("Time to save XlmRoBertaForQuestionAnswering model") {
      pipelineModel.stages.last
        .asInstanceOf[XlmRoBertaForQuestionAnswering]
        .write
        .overwrite()
        .save("./tmp_xlmrobertaforquestion_model")
    }

    val loadedPipelineModel = PipelineModel.load("./tmp_xlmrobertaforquestion_pipeline")
    loadedPipelineModel.transform(ddd).select("label.result").show(false)


  }

  "XlmRoBertaForQuestionAnswering" should "benchmark test" taggedAs SlowTest in {

    val data = ResourceHelper.spark.read
      .option("header", "true")
      .option("escape", "\"")
      .csv("src/test/resources/squad/validation-squad-sample.csv")
      .sample(0.1)

    println(data.count())
    data.show(5)

    val document = new MultiDocumentAssembler()
      .setInputCols("question", "context")
      .setOutputCols("document_question", "document_context")

    val questionAnswering = XlmRoBertaForQuestionAnswering
      .pretrained()
      .setInputCols(Array("document_question", "document_context"))
      .setOutputCol("answer")
      .setCaseSensitive(true)
      .setMaxSentenceLength(512)

    val pipeline = new Pipeline().setStages(Array(document, questionAnswering))

    val pipelineModel = pipeline.fit(data)
    val pipelineDF = pipelineModel.transform(data)

    Benchmark.time("Time to show XlmRoBertaForQuestionAnswering results") {
      pipelineDF.select("answer").show(10, false)
    }

    Benchmark.time("Time to save XlmRoBertaForQuestionAnswering results") {
      pipelineDF
        .select("answer.result")
        .write
        .mode("overwrite")
        .parquet("./tmp_question_answering")
    }

  }
}
