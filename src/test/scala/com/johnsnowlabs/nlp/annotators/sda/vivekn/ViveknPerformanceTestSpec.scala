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

package com.johnsnowlabs.nlp.annotators.sda.vivekn

import com.johnsnowlabs.nlp.annotator._
import com.johnsnowlabs.nlp.base._
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.tags.SlowTest
import com.johnsnowlabs.util.Benchmark
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.functions.rand
import org.scalatest.flatspec.AnyFlatSpec

class ViveknPerformanceTestSpec extends AnyFlatSpec {

  "Vivekn pipeline" should "be fast" taggedAs SlowTest in {

    ResourceHelper.spark
    import ResourceHelper.spark.implicits._

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val sentenceDetector = new SentenceDetector()
      .setInputCols(Array("document"))
      .setOutputCol("sentence")

    val tokenizer = new Tokenizer()
      .setInputCols(Array("document"))
      .setOutputCol("token")

    val vivekn = ViveknSentimentModel.pretrained()
      .setInputCols("document", "token")
      .setOutputCol("sentiment")

    val finisher = new Finisher()
      .setInputCols("sentiment")

    val recursivePipeline = new Pipeline()
      .setStages(Array(
        documentAssembler,
        //sentenceDetector,
        tokenizer,
        vivekn,
        finisher
      ))

    val sentmodel = recursivePipeline.fit(Seq.empty[String].toDF("text"))
    val sentlplight = new LightPipeline(sentmodel)

    Benchmark.time("Positive sentence") {
      println(sentlplight.annotate("Oh man, the movie is so great I can't stop watching it over and over again!!!").values.flatten.mkString(","))
    }
    Benchmark.time("Negative sentence") {
      println(sentlplight.annotate("Don't watch it. It's horrible. You will regret it.").values.flatten.mkString(","))
    }
    Benchmark.time("Known positive") {
      println(sentlplight.annotate("We liked Mission Impossible.").values.flatten.mkString(","))
    }
    Benchmark.time("Known positive") {
      println(sentlplight.annotate("I love Harry Potter..").values.flatten.mkString(","))
    }
    Benchmark.time("Known negative") {
      println(sentlplight.annotate("Brokeback Mountain is fucking horrible..").values.flatten.mkString(","))
    }
    Benchmark.time("Known negative") {
      println(sentlplight.annotate("These Harry Potter movies really suck.").values.flatten.mkString(","))
    }

  }

}
