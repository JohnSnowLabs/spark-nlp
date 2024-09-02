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

import com.johnsnowlabs.nlp.base._
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.tags.SlowTest
import com.johnsnowlabs.util.Benchmark
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.scalatest.flatspec.AnyFlatSpec

class CamemBertForQuestionAnsweringTestSpec extends AnyFlatSpec {

  import ResourceHelper.spark.implicits._

  "CamemBertForQuestionAnswering" should "correctly load custom model with extracted signatures" taggedAs SlowTest in {

    val fquadContext =
      """L'idée selon laquelle une planète inconnue
        |pourrait exister entre les orbites de Mars et
        |Jupiter fut proposée pour la première fois par
        |Johann Elert Bode en 1768. Ses suggestions étaient
        |basées sur la loi de Titius-Bode, une théorie
        |désormais obsolète proposée par Johann Daniel
        |Titius en 1766,. Selon cette loi, le demi-grand
        |axe de cette planète aurait été d'environ 2,8 ua.
        |La découverte d'Uranus par William Herschel en
        |1781 accrut la confiance dans la loi de Titius-
        |Bode et, en 1800, vingt-quatre astronomes
        |expérimentés combinèrent leurs efforts et
        |entreprirent une recherche méthodique de la
        |planète proposée,. Le groupe était dirigé par
        |Franz Xaver von Zach. Bien qu'ils n'aient pas
        |découvert Cérès, ils trouvèrent néanmoins
        |plusieurs autres astéroïdes.""".stripMargin

    val ddd = Seq(
      (
        "Quel astronome a émit l'idée en premier d'une planète entre les orbites de Mars et Jupiter ?",
        fquadContext),
      ("Quel astronome découvrit Uranus ?", fquadContext),
      ("Quelles furent les découvertes finales des vingt-quatre astronomes ?", fquadContext),
      ("Où est-ce que je vis?", "Mon nom est Wolfgang et je vis à Berlin"))
      .toDF("question", "context")
      .repartition(1)

    val document = new MultiDocumentAssembler()
      .setInputCols("question", "context")
      .setOutputCols("document_question", "document_context")

    val questionAnswering = CamemBertForQuestionAnswering
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


  "CamemBertForQuestionAnswering" should "be saved and loaded correctly" taggedAs SlowTest in {

    import ResourceHelper.spark.implicits._

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

    val questionAnswering = CamemBertForQuestionAnswering
      .pretrained()
      .setInputCols(Array("document_question", "document_context"))
      .setOutputCol("answer")
      .setCaseSensitive(false)
      .setMaxSentenceLength(512)


    val pipeline = new Pipeline().setStages(Array(document, questionAnswering))

    val pipelineModel = pipeline.fit(ddd)
    val pipelineDF = pipelineModel.transform(ddd)

    pipelineDF.select("answer.result").show(false)

    Benchmark.time("Time to save CamemBertForQuestionAnswering pipeline model") {
      pipelineModel.write.overwrite().save("./tmp_forquestionanswering_pipeline")
    }

    Benchmark.time("Time to save CamemBertForQuestionAnswering model") {
      pipelineModel.stages.last
        .asInstanceOf[CamemBertForQuestionAnswering]
        .write
        .overwrite()
        .save("./tmp_forquestionanswering_model")
    }

    val loadedPipelineModel = PipelineModel.load("./tmp_forquestionanswering_pipeline")
    loadedPipelineModel.transform(ddd).select("answer.result").show(false)

    val loadedSequenceModel = CamemBertForQuestionAnswering.load("./tmp_forquestionanswering_model")

    }
  "CamemBertForQuestionAnswering" should "benchmark test" taggedAs SlowTest in {

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

    val questionAnswering = CamemBertForQuestionAnswering
      .pretrained()
      .setInputCols(Array("document_question", "document_context"))
      .setOutputCol("answer")
      .setCaseSensitive(true)
      .setMaxSentenceLength(512)

    val pipeline = new Pipeline().setStages(Array(document, questionAnswering))

    val pipelineModel = pipeline.fit(data)
    val pipelineDF = pipelineModel.transform(data)

    Benchmark.time("Time to show CamemBertForQuestionAnswering results") {
      pipelineDF.select("answer").show(10, false)
    }

    Benchmark.time("Time to save CamemBertForQuestionAnswering results") {
      pipelineDF
        .select("answer.result")
        .write
        .mode("overwrite")
        .parquet("./tmp_question_answering")
    }

  }
}
