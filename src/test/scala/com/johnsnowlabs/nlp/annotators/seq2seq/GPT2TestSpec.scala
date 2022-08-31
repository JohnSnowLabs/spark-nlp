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

package com.johnsnowlabs.nlp.annotators.seq2seq

import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.tags.SlowTest
import com.johnsnowlabs.util.Benchmark
import org.apache.spark.ml.Pipeline
import org.scalatest.flatspec.AnyFlatSpec

class GPT2TestSpec extends AnyFlatSpec {
  "gpt2" should "should handle temperature=0 correctly and not crash when predicting more than 1 element with doSample=True" taggedAs SlowTest in {
    // Even tough the Paper states temperature in interval [0,1), using temperature=0 will result in division by 0 error.
    // Also DoSample=True may result in infinities being generated and distFiltered.length==0 which results in exception if we don't return 0 instead internally.
    val testData = ResourceHelper.spark
      .createDataFrame(Seq(
        (1, "My name is Leonardo."),
        (2, "My name is Leonardo and I come from Rome."),
        (3, "My name is"),
        (4, "What is my name?"),
        (5, "Who is Ronaldo?"),
        (6, "Who discovered the microscope?"),
        (7, "Where does petrol come from?"),
        (8, "What is the difference between diesel and petrol?"),
        (9, "Where is Sofia?"),
        (10, "The priest is convinced that")))
      .toDF("id", "text")
      .repartition(1)
    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("documents")

    val gpt2 = GPT2Transformer
      .pretrained()
      .setInputCols(Array("documents"))
      .setMaxOutputLength(50)
      .setDoSample(true)
      .setTopK(50)
      .setTemperature(0)
      .setBatchSize(5)
      .setNoRepeatNgramSize(3)
      .setOutputCol("generation")
    new Pipeline().setStages(Array(documentAssembler, gpt2)).fit(testData).transform(testData).show()

  }

  "gpt2" should "run SparkNLP pipeline with larger batch size" taggedAs SlowTest in {
    val testData = ResourceHelper.spark
      .createDataFrame(Seq(
        (1, "My name is Leonardo."),
        (2, "My name is Leonardo and I come from Rome."),
        (3, "My name is"),
        (4, "What is my name?"),
        (5, "Who is Ronaldo?"),
        (6, "Who discovered the microscope?"),
        (7, "Where does petrol come from?"),
        (8, "What is the difference between diesel and petrol?"),
        (9, "Where is Sofia?"),
        (10, "The priest is convinced that")))
      .toDF("id", "text")
      .repartition(1)

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("documents")

    val gpt2 = GPT2Transformer
      .pretrained()
      .setInputCols(Array("documents"))
      .setMaxOutputLength(50)
      .setDoSample(false)
      .setTopK(50)
      .setBatchSize(5)
      .setNoRepeatNgramSize(3)
      .setOutputCol("generation")

    val pipeline = new Pipeline().setStages(Array(documentAssembler, gpt2))

    val model = pipeline.fit(testData)
    val results = model.transform(testData)

    Benchmark.time("Time to save pipeline the first time", true) {
      results.select("generation.result").write.mode("overwrite").save("./tmp_gpt_pipeline")
    }

    Benchmark.time("Time to save pipeline the second time", true) {
      results.select("generation.result").write.mode("overwrite").save("./tmp_gpt_pipeline")
    }

    Benchmark.time("Time to generate text", true) {
      results.select("generation.result").show(truncate = false)
    }
  }

  "gpt2" should "run SparkNLP pipeline with doSample=true " taggedAs SlowTest in {
    val testData = ResourceHelper.spark
      .createDataFrame(Seq((1, "Leonardo Da Vinci invented the wheel?")))
      .toDF("id", "text")

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("documents")

    val gpt2 = GPT2Transformer
      .pretrained()
      .setTask("Is it true that")
      .setInputCols(Array("documents"))
      .setDoSample(true)
      .setMaxOutputLength(50)
      .setOutputCol("generation")

    val pipeline = new Pipeline().setStages(Array(documentAssembler, gpt2))

    val model = pipeline.fit(testData)

    val dataframe1 = model
      .transform(testData)
      .select("generation.result")
      .collect()
      .toSeq
      .head
      .getAs[Seq[String]](0)
      .head
    println(dataframe1)
    val dataframe2 = model
      .transform(testData)
      .select("generation.result")
      .collect()
      .toSeq
      .head
      .getAs[Seq[String]](0)
      .head
    println(dataframe2)

    assert(!dataframe1.equals(dataframe2))

  }

  "gpt2" should "run SparkNLP pipeline with doSample=true and fixed random seed " taggedAs SlowTest in {
    val testData =
      ResourceHelper.spark.createDataFrame(Seq((1, "Preheat the oven to."))).toDF("id", "text")

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("documents")

    val gpt2 = GPT2Transformer
      .pretrained()
      .setInputCols(Array("documents"))
      .setDoSample(true)
      .setMaxOutputLength(50)
      .setRandomSeed(10)
      .setOutputCol("generation")

    val pipeline = new Pipeline().setStages(Array(documentAssembler, gpt2))

    val model = pipeline.fit(testData)

    val dataframe1 = model
      .transform(testData)
      .select("generation.result")
      .collect()
      .toSeq
      .head
      .getAs[Seq[String]](0)
      .head
    println(dataframe1)
    val dataframe2 = model
      .transform(testData)
      .select("generation.result")
      .collect()
      .toSeq
      .head
      .getAs[Seq[String]](0)
      .head
    println(dataframe2)

    assert(dataframe1.equals(dataframe2))
  }

}
