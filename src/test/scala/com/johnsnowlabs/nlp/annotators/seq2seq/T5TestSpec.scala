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

import com.johnsnowlabs.nlp.annotator.SentenceDetectorDLModel
import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.tags.SlowTest
import com.johnsnowlabs.util.Benchmark
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.functions.col
import org.scalatest.flatspec.AnyFlatSpec

class T5TestSpec extends AnyFlatSpec {

  "google/t5-small-ssm-nq " should "run SparkNLP pipeline" taggedAs SlowTest in {
    val testData = ResourceHelper.spark
      .createDataFrame(
        Seq(
          (1, "Which is the capital of France? Who was the first president of USA?"),
          (1, "Which is the capital of Bulgaria ?"),
          (2, "Who is Donald Trump?")))
      .toDF("id", "text")

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("documents")

    val sentenceDetector = SentenceDetectorDLModel
      .pretrained()
      .setInputCols(Array("documents"))
      .setOutputCol("questions")

    val t5 = T5Transformer
      .pretrained("google_t5_small_ssm_nq")
      .setInputCols(Array("questions"))
      .setOutputCol("answers")

    val pipeline = new Pipeline().setStages(Array(documentAssembler, sentenceDetector, t5))

    val model = pipeline.fit(testData)
    val results = model.transform(testData)

    results.select("questions.result", "answers.result").show(truncate = false)
  }

  "t5-small" should "run SparkNLP pipeline with maxLength=200 " taggedAs SlowTest in {
    val testData = ResourceHelper.spark
      .createDataFrame(
        Seq(
          (
            1,
            "Preheat the oven to 220°C/ fan200°C/gas 7. Trim the lamb fillet of fat and cut into slices the thickness" +
              " of a chop. Cut the kidneys in half and snip out the white core. Melt a knob of dripping or 2 tablespoons " +
              "of vegetable oil in a heavy large pan. Fry the lamb fillet in batches for 3-4 minutes, turning once, until " +
              "browned. Set aside. Fry the kidneys and cook for 1-2 minutes, turning once, until browned. Set aside." +
              "Wipe the pan with kitchen paper, then add the butter. Add the onions and fry for about 10 minutes until " +
              "softened. Sprinkle in the flour and stir well for 1 minute. Gradually pour in the stock, stirring all the " +
              "time to avoid lumps. Add the herbs. Stir the lamb and kidneys into the onions. Season well. Transfer to a" +
              " large 2.5-litre casserole. Slice the peeled potatoes thinly and arrange on top in overlapping rows. Brush " +
              "with melted butter and season. Cover and bake for 30 minutes. Reduce the oven temperature to 160°C" +
              "/fan140°C/gas 3 and cook for a further 2 hours. Then increase the oven temperature to 200°C/ fan180°C/gas 6," +
              " uncover, and brush the potatoes with more butter. Cook uncovered for 15-20 minutes, or until golden.")))
      .toDF("id", "text")

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("documents")

    val t5 = T5Transformer
      .pretrained("t5_small")
      .setTask("summarize:")
      .setInputCols(Array("documents"))
      .setMaxOutputLength(200)
      .setOutputCol("summaries")

    val pipeline = new Pipeline().setStages(Array(documentAssembler, t5))

    val model = pipeline.fit(testData)
    val results = model.transform(testData)

    results.select("summaries.result").show(truncate = false)
    val dataframe = results.select("summaries.result").collect()
    val result = dataframe.toSeq.head.getAs[Seq[String]](0).head

    assert(
      result == "the lamb fillet of fat and cut into slices the thickness of a chop . cut the kidneys in half and snip out the white core .")
  }

  "t5-small" should "run SparkNLP pipeline with doSample=true " taggedAs SlowTest in {
    val testData = ResourceHelper.spark
      .createDataFrame(
        Seq(
          (
            1,
            "Preheat the oven to 220°C/ fan200°C/gas 7. Trim the lamb fillet of fat and cut into slices the thickness" +
              " of a chop. Cut the kidneys in half and snip out the white core. Melt a knob of dripping or 2 tablespoons " +
              "of vegetable oil in a heavy large pan. Fry the lamb fillet in batches for 3-4 minutes, turning once, until " +
              "browned. Set aside. Fry the kidneys and cook for 1-2 minutes, turning once, until browned. Set aside." +
              "Wipe the pan with kitchen paper, then add the butter. Add the onions and fry for about 10 minutes until " +
              "softened. Sprinkle in the flour and stir well for 1 minute. Gradually pour in the stock, stirring all the " +
              "time to avoid lumps. Add the herbs. Stir the lamb and kidneys into the onions. Season well. Transfer to a" +
              " large 2.5-litre casserole. Slice the peeled potatoes thinly and arrange on top in overlapping rows. Brush " +
              "with melted butter and season. Cover and bake for 30 minutes. Reduce the oven temperature to 160°C" +
              "/fan140°C/gas 3 and cook for a further 2 hours. Then increase the oven temperature to 200°C/ fan180°C/gas 6," +
              " uncover, and brush the potatoes with more butter. Cook uncovered for 15-20 minutes, or until golden.")))
      .toDF("id", "text")

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("documents")

    val t5 = T5Transformer
      .pretrained("t5_small")
      .setTask("summarize:")
      .setInputCols(Array("documents"))
      .setDoSample(true)
      .setMaxOutputLength(50)
      .setOutputCol("summaries")

    val pipeline = new Pipeline().setStages(Array(documentAssembler, t5))

    val model = pipeline.fit(testData)
    val dataframe1 = model
      .transform(testData)
      .select("summaries.result")
      .collect()
      .toSeq
      .head
      .getAs[Seq[String]](0)
      .head
    println(dataframe1)
    val dataframe2 = model
      .transform(testData)
      .select("summaries.result")
      .collect()
      .toSeq
      .head
      .getAs[Seq[String]](0)
      .head
    println(dataframe2)

    assert(!dataframe1.equals(dataframe2))
  }

  "t5-small" should "run SparkNLP pipeline with doSample=true and fixed random seed " taggedAs SlowTest in {
    val testData = ResourceHelper.spark
      .createDataFrame(
        Seq(
          (
            1,
            "Preheat the oven to 220°C/ fan200°C/gas 7. Trim the lamb fillet of fat and cut into slices the thickness" +
              " of a chop. Cut the kidneys in half and snip out the white core. Melt a knob of dripping or 2 tablespoons " +
              "of vegetable oil in a heavy large pan. Fry the lamb fillet in batches for 3-4 minutes, turning once, until " +
              "browned. Set aside. Fry the kidneys and cook for 1-2 minutes, turning once, until browned. Set aside." +
              "Wipe the pan with kitchen paper, then add the butter. Add the onions and fry for about 10 minutes until " +
              "softened. Sprinkle in the flour and stir well for 1 minute. Gradually pour in the stock, stirring all the " +
              "time to avoid lumps. Add the herbs. Stir the lamb and kidneys into the onions. Season well. Transfer to a" +
              " large 2.5-litre casserole. Slice the peeled potatoes thinly and arrange on top in overlapping rows. Brush " +
              "with melted butter and season. Cover and bake for 30 minutes. Reduce the oven temperature to 160°C" +
              "/fan140°C/gas 3 and cook for a further 2 hours. Then increase the oven temperature to 200°C/ fan180°C/gas 6," +
              " uncover, and brush the potatoes with more butter. Cook uncovered for 15-20 minutes, or until golden.")))
      .toDF("id", "text")

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("documents")

    val t5 = T5Transformer
      .pretrained("t5_small")
      .setTask("summarize:")
      .setInputCols(Array("documents"))
      .setDoSample(true)
      .setMaxOutputLength(50)
      .setRandomSeed(10L)
      .setOutputCol("summaries")

    val pipeline = new Pipeline().setStages(Array(documentAssembler, t5))

    val model = pipeline.fit(testData)

    val dataframe1 = model
      .transform(testData)
      .select("summaries.result")
      .collect()
      .toSeq
      .head
      .getAs[Seq[String]](0)
      .head
    println(dataframe1)
    val dataframe2 = model
      .transform(testData)
      .select("summaries.result")
      .collect()
      .toSeq
      .head
      .getAs[Seq[String]](0)
      .head
    println(dataframe2)

    assert(dataframe1.equals(dataframe2))
  }

  "t5-small" should "run SparkNLP pipeline with doSample=true, fixed random seed deactivated topK" taggedAs SlowTest in {
    val testData = ResourceHelper.spark
      .createDataFrame(
        Seq(
          (
            1,
            "Preheat the oven to 220°C/ fan200°C/gas 7. Trim the lamb fillet of fat and cut into slices the thickness" +
              " of a chop. Cut the kidneys in half and snip out the white core. Melt a knob of dripping or 2 tablespoons " +
              "of vegetable oil in a heavy large pan. Fry the lamb fillet in batches for 3-4 minutes, turning once, until " +
              "browned. Set aside. Fry the kidneys and cook for 1-2 minutes, turning once, until browned. Set aside." +
              "Wipe the pan with kitchen paper, then add the butter. Add the onions and fry for about 10 minutes until " +
              "softened. Sprinkle in the flour and stir well for 1 minute. Gradually pour in the stock, stirring all the " +
              "time to avoid lumps. Add the herbs. Stir the lamb and kidneys into the onions. Season well. Transfer to a" +
              " large 2.5-litre casserole. Slice the peeled potatoes thinly and arrange on top in overlapping rows. Brush " +
              "with melted butter and season. Cover and bake for 30 minutes. Reduce the oven temperature to 160°C" +
              "/fan140°C/gas 3 and cook for a further 2 hours. Then increase the oven temperature to 200°C/ fan180°C/gas 6," +
              " uncover, and brush the potatoes with more butter. Cook uncovered for 15-20 minutes, or until golden.")))
      .toDF("id", "text")

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("documents")

    val t5 = T5Transformer
      .pretrained("t5_small")
      .setTask("summarize:")
      .setInputCols(Array("documents"))
      .setOutputCol("summaries")
      .setDoSample(true)
      .setRandomSeed(10L)
      .setMaxOutputLength(20)
      .setTopK(0)

    val pipeline = new Pipeline().setStages(Array(documentAssembler, t5))

    val model = pipeline.fit(testData)
    val results1 = model.transform(testData)

    val dataframe1 =
      results1.select("summaries.result").collect().toSeq.head.getAs[Seq[String]](0).head
    assert(
      dataframe1 == "cook 2 months uncovered and uncovered for 15-20 mins with more butter . heat over medium")

  }

  "t5-small" should "run SparkNLP pipeline with temperature to decrease the sensitivity to low probability candidates" taggedAs SlowTest in {
    val testData = ResourceHelper.spark
      .createDataFrame(
        Seq(
          (
            1,
            "Preheat the oven to 220°C/ fan200°C/gas 7. Trim the lamb fillet of fat and cut into slices the thickness" +
              " of a chop. Cut the kidneys in half and snip out the white core. Melt a knob of dripping or 2 tablespoons " +
              "of vegetable oil in a heavy large pan. Fry the lamb fillet in batches for 3-4 minutes, turning once, until " +
              "browned. Set aside. Fry the kidneys and cook for 1-2 minutes, turning once, until browned. Set aside." +
              "Wipe the pan with kitchen paper, then add the butter. Add the onions and fry for about 10 minutes until " +
              "softened. Sprinkle in the flour and stir well for 1 minute. Gradually pour in the stock, stirring all the " +
              "time to avoid lumps. Add the herbs. Stir the lamb and kidneys into the onions. Season well. Transfer to a" +
              " large 2.5-litre casserole. Slice the peeled potatoes thinly and arrange on top in overlapping rows. Brush " +
              "with melted butter and season. Cover and bake for 30 minutes. Reduce the oven temperature to 160°C" +
              "/fan140°C/gas 3 and cook for a further 2 hours. Then increase the oven temperature to 200°C/ fan180°C/gas 6," +
              " uncover, and brush the potatoes with more butter. Cook uncovered for 15-20 minutes, or until golden.")))
      .toDF("id", "text")

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("documents")

    val t5 = T5Transformer
      .pretrained("t5_small")
      .setTask("summarize:")
      .setInputCols(Array("documents"))
      .setOutputCol("summaries")
      .setDoSample(true)
      .setRandomSeed(10L)
      .setMaxOutputLength(50)
      .setTemperature(0.7)
      .setTopK(50)

    val pipeline = new Pipeline().setStages(Array(documentAssembler, t5))

    val model = pipeline.fit(testData)
    val results1 = model.transform(testData)

    val dataframe1 =
      results1.select("summaries.result").collect().toSeq.head.getAs[Seq[String]](0).head
    println(dataframe1)
    assert(
      "dripping or 2 tablespoons of vegetable oil set aside, stirring constantly . add the onions and fry for about 10 minutes until softened ." == dataframe1)

  }

  "t5-small" should "run SparkNLP pipeline with doSample and TopP" taggedAs SlowTest in {
    val testData = ResourceHelper.spark
      .createDataFrame(
        Seq(
          (
            1,
            "Preheat the oven to 220°C/ fan200°C/gas 7. Trim the lamb fillet of fat and cut into slices the thickness" +
              " of a chop. Cut the kidneys in half and snip out the white core. Melt a knob of dripping or 2 tablespoons " +
              "of vegetable oil in a heavy large pan. Fry the lamb fillet in batches for 3-4 minutes, turning once, until " +
              "browned. Set aside. Fry the kidneys and cook for 1-2 minutes, turning once, until browned. Set aside." +
              "Wipe the pan with kitchen paper, then add the butter. Add the onions and fry for about 10 minutes until " +
              "softened. Sprinkle in the flour and stir well for 1 minute. Gradually pour in the stock, stirring all the " +
              "time to avoid lumps. Add the herbs. Stir the lamb and kidneys into the onions. Season well. Transfer to a" +
              " large 2.5-litre casserole. Slice the peeled potatoes thinly and arrange on top in overlapping rows. Brush " +
              "with melted butter and season. Cover and bake for 30 minutes. Reduce the oven temperature to 160°C" +
              "/fan140°C/gas 3 and cook for a further 2 hours. Then increase the oven temperature to 200°C/ fan180°C/gas 6," +
              " uncover, and brush the potatoes with more butter. Cook uncovered for 15-20 minutes, or until golden.")))
      .toDF("id", "text")

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("documents")

    val t5 = T5Transformer
      .pretrained("t5_small")
      .setTask("summarize:")
      .setInputCols(Array("documents"))
      .setOutputCol("summaries")
      .setDoSample(true)
      .setRandomSeed(10L)
      .setMaxOutputLength(50)
      .setTopP(0.7)
      .setTopK(0)

    val pipeline = new Pipeline().setStages(Array(documentAssembler, t5))

    val model = pipeline.fit(testData)
    val results1 = model.transform(testData)

    val dataframe1 =
      results1.select("summaries.result").collect().toSeq.head.getAs[Seq[String]](0).head
    println(dataframe1)
    assert(
      "the lamb fillet is cut into slices the thickness of a chop . add the kidneys and cook for 1-2 minutes, turning once, until browned ." == dataframe1)

  }

  "t5-small" should "run SparkNLP pipeline with repetitionPenalty" taggedAs SlowTest in {
    val testData = ResourceHelper.spark
      .createDataFrame(
        Seq(
          (
            1,
            "Preheat the oven to 220°C/ fan200°C/gas 7. Trim the lamb fillet of fat and cut into slices the thickness" +
              " of a chop. Cut the kidneys in half and snip out the white core. Melt a knob of dripping or 2 tablespoons " +
              "of vegetable oil in a heavy large pan. Fry the lamb fillet in batches for 3-4 minutes, turning once, until " +
              "browned. Set aside. Fry the kidneys and cook for 1-2 minutes, turning once, until browned. Set aside." +
              "Wipe the pan with kitchen paper, then add the butter. Add the onions and fry for about 10 minutes until " +
              "softened. Sprinkle in the flour and stir well for 1 minute. Gradually pour in the stock, stirring all the " +
              "time to avoid lumps. Add the herbs. Stir the lamb and kidneys into the onions. Season well. Transfer to a" +
              " large 2.5-litre casserole. Slice the peeled potatoes thinly and arrange on top in overlapping rows. Brush " +
              "with melted butter and season. Cover and bake for 30 minutes. Reduce the oven temperature to 160°C" +
              "/fan140°C/gas 3 and cook for a further 2 hours. Then increase the oven temperature to 200°C/ fan180°C/gas 6," +
              " uncover, and brush the potatoes with more butter. Cook uncovered for 15-20 minutes, or until golden.")))
      .toDF("id", "text")

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("documents")

    val t5 = T5Transformer
      .pretrained("t5_small")
      .setTask("summarize:")
      .setInputCols(Array("documents"))
      .setOutputCol("summaries")
      .setDoSample(false)
      .setMaxOutputLength(50)
      .setRepetitionPenalty(2)

    val pipeline = new Pipeline().setStages(Array(documentAssembler, t5))

    val model = pipeline.fit(testData)
    val results1 = model.transform(testData)

    val dataframe1 =
      results1.select("summaries.result").collect().toSeq.head.getAs[Seq[String]](0).head
    println(dataframe1)

    assert(
      dataframe1 == "the lamb fillet of fat and cut into slices the thickness of a chop . heat up to 220°C/fan140°C/gas 7 and cook for another 2 hours - uncover, and brush the potatoes with more")

  }

  "t5-small" should "run SparkNLP pipeline and ignore a token" taggedAs SlowTest in {
    val testData = ResourceHelper.spark
      .createDataFrame(Seq(
        (
          1,
          "Preheat the oven to 220°C/ fan200°C/gas 7. Trim the lamb fillet of fat and cut into slices the thickness" +
            " of a chop. Cut the kidneys in half and snip out the white core. Melt a knob of dripping or 2 tablespoons " +
            "of vegetable oil in a heavy large pan. Fry the lamb fillet in batches for 3-4 minutes, turning once, until " +
            "browned. Set aside. Fry the kidneys and cook for 1-2 minutes, turning once, until browned. Set aside." +
            "Wipe the pan with kitchen paper, then add the butter. Add the onions and fry for about 10 minutes until " +
            "softened. Sprinkle in the flour and stir well for 1 minute. Gradually pour in the stock, stirring all the " +
            "time to avoid lumps. Add the herbs. Stir the lamb and kidneys into the onions. Season well. Transfer to a" +
            " large 2.5-litre casserole. Slice the peeled potatoes thinly and arrange on top in overlapping rows. Brush " +
            "with melted butter and season. Cover and bake for 30 minutes. Reduce the oven temperature to 160°C" +
            "/fan140°C/gas 3 and cook for a further 2 hours. Then increase the oven temperature to 200°C/ fan180°C/gas 6," +
            " uncover, and brush the potatoes with more butter. Cook uncovered for 15-20 minutes, or until golden."),
        (
          1,
          "Donald John Trump (born June 14, 1946) is the 45th and current president of the United States. Before " +
            "entering politics, he was a businessman and television personality. Born and raised in Queens, New York " +
            "City, Trump attended Fordham University for two years and received a bachelor's degree in economics from the " +
            "Wharton School of the University of Pennsylvania. He became president of his father Fred Trump's real " +
            "estate business in 1971, renamed it The Trump Organization, and expanded its operations to building or " +
            "renovating skyscrapers, hotels, casinos, and golf courses. Trump later started various side ventures," +
            " mostly by licensing his name. Trump and his businesses have been involved in more than 4,000 state and" +
            " federal legal actions, including six bankruptcies. He owned the Miss Universe brand of beauty pageants " +
            "from 1996 to 2015, and produced and hosted the reality television series The Apprentice from 2004 to 2015.")))
      .toDF("id", "text")

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("documents")

    val t5 = T5Transformer
      .pretrained("t5_small")
      .setTask("summarize:")
      .setInputCols(Array("documents"))
      .setMaxOutputLength(200)
      .setIgnoreTokenIds(Array(12065)) // ignore token "vegetable"
      .setOutputCol("summaries")

    val pipeline = new Pipeline().setStages(Array(documentAssembler, t5))

    val model = pipeline.fit(testData)
    val results = model.transform(testData)

    Benchmark.time("Time to save pipeline the first time") {
      results.select("summaries.result").write.mode("overwrite").save("./tmp_t5_pipeline")
    }

    Benchmark.time("Time to save pipeline the second time") {
      results.select("summaries.result").write.mode("overwrite").save("./tmp_t5_pipeline")
    }

    results.select("summaries.result").show(truncate = false)

    assert(
      results
        .selectExpr("explode(summaries) AS summary")
        .where(col("summary.result").contains(" vegetable "))
        .count() == 0,
      "should not include ignored tokens")
  }

  "t5-small" should "run SparkNLP pipeline for translation" taggedAs SlowTest in {
    val testData = ResourceHelper.spark
      .createDataFrame(Seq(
        (
          1,
          "Preheat the oven to 220°C/ fan200°C/gas 7. Trim the lamb fillet of fat and cut into slices the thickness" +
            " of a chop. Cut the kidneys in half and snip out the white core. Melt a knob of dripping or 2 tablespoons " +
            "of vegetable oil in a heavy large pan. Fry the lamb fillet in batches for 3-4 minutes, turning once, until " +
            "browned. Set aside. Fry the kidneys and cook for 1-2 minutes, turning once, until browned. Set aside." +
            "Wipe the pan with kitchen paper, then add the butter. Add the onions and fry for about 10 minutes until " +
            "softened. Sprinkle in the flour and stir well for 1 minute. Gradually pour in the stock, stirring all the " +
            "time to avoid lumps. Add the herbs. Stir the lamb and kidneys into the onions. Season well. Transfer to a" +
            " large 2.5-litre casserole. Slice the peeled potatoes thinly and arrange on top in overlapping rows. Brush " +
            "with melted butter and season. Cover and bake for 30 minutes. Reduce the oven temperature to 160°C" +
            "/fan140°C/gas 3 and cook for a further 2 hours. Then increase the oven temperature to 200°C/ fan180°C/gas 6," +
            " uncover, and brush the potatoes with more butter. Cook uncovered for 15-20 minutes, or until golden."),
        (
          1,
          "Donald John Trump (born June 14, 1946) is the 45th and current president of the United States. Before " +
            "entering politics, he was a businessman and television personality. Born and raised in Queens, New York " +
            "City, Trump attended Fordham University for two years and received a bachelor's degree in economics from the " +
            "Wharton School of the University of Pennsylvania. He became president of his father Fred Trump's real " +
            "estate business in 1971, renamed it The Trump Organization, and expanded its operations to building or " +
            "renovating skyscrapers, hotels, casinos, and golf courses. Trump later started various side ventures," +
            " mostly by licensing his name. Trump and his businesses have been involved in more than 4,000 state and" +
            " federal legal actions, including six bankruptcies. He owned the Miss Universe brand of beauty pageants " +
            "from 1996 to 2015, and produced and hosted the reality television series The Apprentice from 2004 to 2015.")))
      .toDF("id", "text")

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("documents")

    val t5 = T5Transformer
      .pretrained("t5_small")
      .setTask("summarize:")
      .setInputCols(Array("documents"))
      .setMaxOutputLength(200)
      .setIgnoreTokenIds(Array(12065)) // ignore token "vegetable"
      .setOutputCol("summaries")

    val pipeline = new Pipeline().setStages(Array(documentAssembler, t5))

    val model = pipeline.fit(testData)
    val results = model.transform(testData).cache()

    Benchmark.time("Time to save pipeline the first time") {
      results.select("summaries.result").write.mode("overwrite").save("./tmp_t5_pipeline")
    }

    Benchmark.time("Time to save pipeline the second time") {
      results.select("summaries.result").write.mode("overwrite").save("./tmp_t5_pipeline")
    }

    assert(
      results
        .selectExpr("explode(summaries) AS summary")
        .where(col("summary.result").contains(" vegetable "))
        .count() == 0,
      "should not include ignored tokens")
  }

  "Pretrained models" should "able to change task" taggedAs SlowTest in {
    val testData =
      ResourceHelper.spark.createDataFrame(Seq((1, "That is good."))).toDF("id", "text")

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("documents")

    val t5 = T5Transformer
      .pretrained("t5_small")
      .setTask("translate:")
      .setInputCols(Array("documents"))
      .setMaxOutputLength(200)
      .setOutputCol("translations")

    t5.setTask("translate English to German:")

    val pipeline = new Pipeline().setStages(Array(documentAssembler, t5))

    val model = pipeline.fit(testData)
    val results = model.transform(testData).cache()

    val collected = results.selectExpr("explode(translations.result)").collect().head.getString(0)
    val expected = "Das ist gut."
    assert(collected == expected, "translation should be correct")
  }

  "Pretrained models" should "be saved and loaded correctly" taggedAs SlowTest in {
    val testData =
      ResourceHelper.spark.createDataFrame(Seq((1, "That is good."))).toDF("id", "text")

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("documents")

    val t5 = T5Transformer
      .pretrained("t5_small")
      .setTask("translate:")
      .setInputCols(Array("documents"))
      .setMaxOutputLength(200)
      .setOutputCol("translations")

    t5.setTask("translate English to German:")

    val pipeline = new Pipeline().setStages(Array(documentAssembler, t5))

    val model = pipeline.fit(testData)
    val results = model.transform(testData).cache()

    results.selectExpr("explode(translations.result)").collect().head.getString(0)

    model.stages.last.asInstanceOf[T5Transformer].write.overwrite().save("./tmp_t5_model")

    val t5Loaded = T5Transformer.load("./tmp_t5_model")

    val pipeline2 = new Pipeline().setStages(Array(documentAssembler, t5Loaded))
    val model2 = pipeline2.fit(testData)
    val results2 = model2.transform(testData)

    results2.select("documents.result", "translations.result").show(truncate = false)
  }

}
