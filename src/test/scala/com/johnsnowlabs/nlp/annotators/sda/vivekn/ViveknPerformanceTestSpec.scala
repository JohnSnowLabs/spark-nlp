/*
 * Copyright 2017-2021 John Snow Labs
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

package com.johnsnowlabs.nlp.annotators.sda.vivekn

import com.johnsnowlabs.nlp.annotator._
import com.johnsnowlabs.nlp.base._
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.tags.SlowTest
import com.johnsnowlabs.util.Benchmark
import org.apache.spark.ml.PipelineModel
import org.apache.spark.sql.functions.rand
import org.scalatest._

class ViveknPerformanceTestSpec extends FlatSpec {

  "Vivekn pipeline" should "be fast" taggedAs SlowTest in {

    ResourceHelper.spark
    import ResourceHelper.spark.implicits._

    val documentAssembler = new DocumentAssembler().
      setInputCol("text").
      setOutputCol("document")

    val sentenceDetector = new SentenceDetector().
      setInputCols(Array("document")).
      setOutputCol("sentence")

    val tokenizer = new Tokenizer().
      setInputCols(Array("document")).
      setOutputCol("token")

    val vivekn = ViveknSentimentModel.load("./my_models/vivekn_opt/").
      setInputCols("document", "token").
      setOutputCol("sentiment")

    val finisher = new Finisher().
      setInputCols("sentiment")

    val recursivePipeline = new RecursivePipeline().
      setStages(Array(
        documentAssembler,
        //sentenceDetector,
        tokenizer,
        vivekn,
        finisher
      ))

    val sentmodel = recursivePipeline.fit(Seq.empty[String].toDF("text"))
    val sentlplight = new LightPipeline(sentmodel)

    val n = 50000

    val parquet = ResourceHelper.spark.read
      .text("./training_positive")
      .toDF("text").sort(rand())
    val data = parquet.as[String].take(n)
    data.length

    val r = Benchmark.time("Light annotate sentiment") {sentlplight.annotate(data)}
    println("positive ratio", r.flatMap(_.values).flatten.count(_ == "positive") / data.length.toDouble)
    println("negative ratio", r.flatMap(_.values).flatten.count(_ == "negative") / data.length.toDouble)
    println("na ratio", r.flatMap(_.values).flatten.count(_ == "na") / data.length.toDouble)

    Benchmark.time("Positive sentence") {println(sentlplight.annotate("Oh man, the movie is so great I can't stop watching it over and over again!!!").values.flatten.mkString(","))}
    Benchmark.time("Negative sentence") {println(sentlplight.annotate("Don't watch it. It's horrible. You will regret it.").values.flatten.mkString(","))}
    Benchmark.time("Known positive") {println(sentlplight.annotate("We liked Mission Impossible.").values.flatten.mkString(","))}
    Benchmark.time("Known positive") {println(sentlplight.annotate("I love Harry Potter..").values.flatten.mkString(","))}
    Benchmark.time("Known negative") {println(sentlplight.annotate("Brokeback Mountain is fucking horrible..").values.flatten.mkString(","))}
    Benchmark.time("Known negative") {println(sentlplight.annotate("These Harry Potter movies really suck.").values.flatten.mkString(","))}

  }

  "Vivekn pipeline with spell checker" should "be fast" taggedAs SlowTest in {

    ResourceHelper.spark
    import ResourceHelper.spark.implicits._

    val documentAssembler = new DocumentAssembler().
      setInputCol("text").
      setOutputCol("document")

    val tokenizer = new Tokenizer().
      setInputCols(Array("document")).
      setOutputCol("token")

    val spell = SymmetricDeleteModel.pretrained()
      .setInputCols("token")
      .setOutputCol("spell")

    val vivekn = ViveknSentimentModel.load("./my_models/vivekn_opt/").
      setInputCols("document", "spell").
      setOutputCol("sentiment")

    val finisher = new Finisher().
      setInputCols("sentiment")

    val recursivePipeline = new RecursivePipeline().
      setStages(Array(
        documentAssembler,
        tokenizer,
        spell,
        vivekn,
        finisher
      ))

    val sentmodel = recursivePipeline.fit(Seq.empty[String].toDF("text"))
    val sentlplight = new LightPipeline(sentmodel)

    val n = 2000

    val parquet = ResourceHelper.spark.read
      .text("./vivekn/training_positive")
      .toDF("text").sort(rand())
    val data = parquet.as[String].take(n)
    println(s"Data size is ${data.length}")

    val sentpipsym = new LightPipeline(PipelineModel.load("./my_models/pipeline_vivekn_sym/"))

    val sentpip = new LightPipeline(PipelineModel.load("./my_models/pipeline_vivekn/"))

    Benchmark.time("Sentiment pipeline with Symmetric Spell Checker no Normalizer") {sentlplight.annotate(data)}

    Benchmark.time("Sentiment pipeline with Symmetric Spell Checker and Normalizer") {sentpipsym.annotate(data)}

    Benchmark.time("Sentiment pipeline with Norvig Spell Checker and Normalizer") {sentpip.annotate(data)}

  }

}
