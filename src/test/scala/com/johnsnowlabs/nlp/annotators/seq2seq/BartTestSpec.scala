/*
 * Copyright 2017-2023 John Snow Labs
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

import com.johnsnowlabs.nlp.annotator.Tokenizer
import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.embeddings.XlmRoBertaSentenceEmbeddings
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.tags.{FastTest, SlowTest}
import com.johnsnowlabs.util.Benchmark
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.scalatest.flatspec.AnyFlatSpec

class BartTestSpec extends AnyFlatSpec {

  "bart-large-cnn" should "should handle temperature=0 correctly and not crash when predicting more than 1 element with doSample=True" taggedAs SlowTest in {
    // Even tough the Paper states temperature in interval [0,1), using temperature=0 will result in division by 0 error.
    // Also DoSample=True may result in infinities being generated and distFiltered.length==0 which results in exception if we don't return 0 instead internally.
    val testData = ResourceHelper.spark
      .createDataFrame(
        Seq(
          (
            1,
            "PG&E stated it scheduled the blackouts in response to forecasts for high winds " +
              "amid dry conditions. The aim is to reduce the risk of wildfires. Nearly 800 thousand customers were " +
              "scheduled to be affected by the shutoffs which were expected to last through at least midday tomorrow.")))
      .toDF("id", "text")
      .repartition(1)
    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("documents")

    val bart = BartTransformer
      .pretrained("distilbart_xsum_12_6")
      .setTask("summarize:")
      .setInputCols(Array("documents"))
      .setDoSample(true)
      .setMaxOutputLength(30)
      .setOutputCol("generation")
    new Pipeline()
      .setStages(Array(documentAssembler, bart))
      .fit(testData)
      .transform(testData)
      .show(truncate = false)

  }

  "distilbart_xsum_12_6" should "download, save, and load a model" taggedAs SlowTest in {

    import ResourceHelper.spark.implicits._

    val ddd = Seq("Something is weird on the notebooks, something is happening.").toDF("text")

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("documents")

    val bart = BartTransformer
      .pretrained("distilbart_xsum_12_6")
      .setTask("summarize:")
      .setInputCols(Array("documents"))
      .setDoSample(true)
      .setMaxOutputLength(30)
      .setOutputCol("generation")

    val pipeline = new Pipeline().setStages(Array(documentAssembler, bart)).fit(ddd)

    pipeline.write.overwrite().save("./tmp_bart_transformer_pipeline")
    val pipelineModel = PipelineModel.load("./tmp_bart_transformer_pipeline")

    pipeline
      .stages(1)
      .asInstanceOf[BartTransformer]
      .write
      .overwrite()
      .save("./tmp_bart_transformer_model")

    pipelineModel.transform(ddd).show()
  }
  "distilbart_xsum_12_6" should "handle text inputs longer than 512 and not crash" taggedAs SlowTest in {
    // text longer than 512
    val testData = ResourceHelper.spark
      .createDataFrame(
        Seq(
          (
            1,
            "PG&E stated it scheduled the blackouts in response to forecasts for high winds " +
              "amid dry conditions. The aim is to reduce the risk of wildfires. Nearly 800 thousand customers were " +
              "scheduled to be affected by the shutoffs which were expected to last through at least midday tomorrow." +
              "PG&E stated it scheduled the blackouts in response to forecasts for high winds " +
              "amid dry conditions. The aim is to reduce the risk of wildfires. Nearly 800 thousand customers were " +
              "scheduled to be affected by the shutoffs which were expected to last through at least midday tomorrow." +
              "PG&E stated it scheduled the blackouts in response to forecasts for high winds " +
              "amid dry conditions. The aim is to reduce the risk of wildfires. Nearly 800 thousand customers were " +
              "scheduled to be affected by the shutoffs which were expected to last through at least midday tomorrow." +
              "PG&E stated it scheduled the blackouts in response to forecasts for high winds " +
              "amid dry conditions. The aim is to reduce the risk of wildfires. Nearly 800 thousand customers were " +
              "scheduled to be affected by the shutoffs which were expected to last through at least midday tomorrow." +
              "PG&E stated it scheduled the blackouts in response to forecasts for high winds " +
              "amid dry conditions. The aim is to reduce the risk of wildfires. Nearly 800 thousand customers were " +
              "scheduled to be affected by the shutoffs which were expected to last through at least midday tomorrow." +
              "PG&E stated it scheduled the blackouts in response to forecasts for high winds " +
              "amid dry conditions. The aim is to reduce the risk of wildfires. Nearly 800 thousand customers were " +
              "scheduled to be affected by the shutoffs which were expected to last through at least midday tomorrow." +
              "PG&E stated it scheduled the blackouts in response to forecasts for high winds " +
              "amid dry conditions. The aim is to reduce the risk of wildfires. Nearly 800 thousand customers were " +
              "scheduled to be affected by the shutoffs which were expected to last through at least midday tomorrow." +
              "PG&E stated it scheduled the blackouts in response to forecasts for high winds " +
              "amid dry conditions. The aim is to reduce the risk of wildfires. Nearly 800 thousand customers were " +
              "scheduled to be affected by the shutoffs which were expected to last through at least midday tomorrow." +
              "PG&E stated it scheduled the blackouts in response to forecasts for high winds " +
              "amid dry conditions. The aim is to reduce the risk of wildfires. Nearly 800 thousand customers were " +
              "scheduled to be affected by the shutoffs which were expected to last through at least midday tomorrow." +
              "PG&E stated it scheduled the blackouts in response to forecasts for high winds " +
              "amid dry conditions. The aim is to reduce the risk of wildfires. Nearly 800 thousand customers were " +
              "scheduled to be affected by the shutoffs which were expected to last through at least midday tomorrow." +
              "PG&E stated it scheduled the blackouts in response to forecasts for high winds " +
              "amid dry conditions. The aim is to reduce the risk of wildfires. Nearly 800 thousand customers were " +
              "scheduled to be affected by the shutoffs which were expected to last through at least midday tomorrow." +
              "PG&E stated it scheduled the blackouts in response to forecasts for high winds " +
              "amid dry conditions. The aim is to reduce the risk of wildfires. Nearly 800 thousand customers were " +
              "scheduled to be affected by the shutoffs which were expected to last through at least midday tomorrow." +
              "PG&E stated it scheduled the blackouts in response to forecasts for high winds " +
              "amid dry conditions. The aim is to reduce the risk of wildfires. Nearly 800 thousand customers were " +
              "scheduled to be affected by the shutoffs which were expected to last through at least midday tomorrow." +
              "PG&E stated it scheduled the blackouts in response to forecasts for high winds " +
              "amid dry conditions. The aim is to reduce the risk of wildfires. Nearly 800 thousand customers were " +
              "scheduled to be affected by the shutoffs which were expected to last through at least midday tomorrow.")))
      .toDF("id", "text")

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("documents")

    val bart = BartTransformer
      .pretrained("distilbart_xsum_12_6")
      .setTask("summarize:")
      .setInputCols(Array("documents"))
      .setDoSample(true)
      .setMaxOutputLength(30)
      .setOutputCol("generation")

    new Pipeline()
      .setStages(Array(documentAssembler, bart))
      .fit(testData)
      .transform(testData)
      .select("generation.result")
      .show(truncate = false)
  }

  "bart-large-cnn" should "run SparkNLP pipeline with maxLength=130 and doSample=true" taggedAs SlowTest in {
    val testData = ResourceHelper.spark
      .createDataFrame(
        Seq((
          1,
          "New York (CNN)When Liana Barrientos was 23 years old, she got married in Westchester County, New York.\nA " +
            "year later, she got married again in Westchester County, but to a different man and without divorcing her first husband." +
            "\nOnly 18 days after that marriage, she got hitched yet again. Then, Barrientos declared \"I do\" five more times, sometimes " +
            "only within two weeks of each other.\nIn 2010, she married once more, this time in the Bronx. In an application for a marriage " +
            "license, she stated it was her \"first and only\" marriage.\nBarrientos, now 39, is facing two criminal counts of \"offering a " +
            "false instrument for filing in the first degree,\" referring to her false statements on the\n2010 marriage license application, " +
            "according to court documents.\nProsecutors said the marriages were part of an immigration scam.\nOn Friday, she pleaded not guilty " +
            "at State Supreme Court in the Bronx, according to her attorney, Christopher Wright, who declined to comment further.\nAfter leaving court, " +
            "Barrientos was arrested and charged with theft of service and criminal trespass for allegedly sneaking into the New York subway through an " +
            "emergency exit, said Detective\nAnnette Markowski, a police spokeswoman. In total, Barrientos has been married 10 times, with nine of her " +
            "marriages occurring between 1999 and 2002.\nAll occurred either in Westchester County, Long Island, New Jersey or the Bronx. She is believed to" +
            " still be married to four men, and at one time, she was married to eight men at once, prosecutors say.\nProsecutors said the immigration scam involved " +
            "some of her husbands, who filed for permanent residence status shortly after the marriages.\nAny divorces happened only after such filings were" +
            " approved. It was unclear whether any of the men will be prosecuted.\nThe case was referred to the Bronx District Attorney\\'s Office " +
            "by Immigration and Customs Enforcement and the Department of Homeland Security\\'s\nInvestigation Division. Seven of the men are from so-called" +
            " \"red-flagged\" countries, including Egypt, Turkey, Georgia, Pakistan and Mali.")))
      .toDF("id", "text")

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("documents")

    val bart = BartTransformer
      .pretrained("distilbart_xsum_12_6")
      .setTask("summarize:")
      .setInputCols(Array("documents"))
      .setMaxOutputLength(70)
      .setMinOutputLength(30)
      .setDoSample(true)
      .setOutputCol("summaries")

    val pipeline = new Pipeline().setStages(Array(documentAssembler, bart))

    val model = pipeline.fit(testData)
    val results = model.transform(testData)

    results.select("summaries.result").show(truncate = false)
//    val dataframe = results.select("summaries.result").collect()
//    val result = dataframe.toSeq.head.getAs[Seq[String]](0).head
//    println(result)
    //    assert(
    //      result == "a knob of dripping or 2 tablespoons of vegetable oil in a large large pan . cut the kidneys in half and snip out the white core . heat the pan for 1-2 minutes, turning once, until browned .")
  }
  "bart-large-cnn" should "run SparkNLP pipeline with maxLength=100 " taggedAs SlowTest in {
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

    val bart = BartTransformer
      .pretrained("distilbart_xsum_12_6")
      .setTask("summarize:")
      .setInputCols(Array("documents"))
      .setMaxOutputLength(100)
      .setOutputCol("summaries")

    val pipeline = new Pipeline().setStages(Array(documentAssembler, bart))

    val model = pipeline.fit(testData)
    val results = model.transform(testData)

    results.select("summaries.result").show(truncate = false)
    val dataframe = results.select("summaries.result").collect()
    val result = dataframe.toSeq.head.getAs[Seq[String]](0).head
    println(result)
//    assert(
//      result == "a knob of dripping or 2 tablespoons of vegetable oil in a large large pan . cut the kidneys in half and snip out the white core . heat the pan for 1-2 minutes, turning once, until browned .")
  }
  "bart-large-cnn" should "run SparkNLP pipeline with doSample=true " taggedAs SlowTest in {
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

    val bart = BartTransformer
      .pretrained("distilbart_xsum_12_6")
      .setTask("summarize:")
      .setInputCols(Array("documents"))
      .setDoSample(true)
      .setRandomSeed(56)
      .setMaxOutputLength(50)
      .setOutputCol("summaries")

    val pipeline = new Pipeline().setStages(Array(documentAssembler, bart))

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

  "bart-large-cnn" should "run SparkNLP pipeline with doSample=false and later change to true " taggedAs SlowTest in {
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

    val bart = BartTransformer
      .pretrained("distilbart_xsum_12_6")
      .setTask("summarize:")
      .setInputCols(Array("documents"))
      .setDoSample(false)
      .setRandomSeed(56)
      .setMaxOutputLength(128)
      .setTemperature(0.1)
      .setOutputCol("summaries")

    val pipeline = new Pipeline().setStages(Array(documentAssembler, bart))

    val model = pipeline.fit(testData)

    var dataframe1 = model
      .transform(testData)
      .select("summaries.result")
      .collect()
      .toSeq
      .head
      .getAs[Seq[String]](0)
      .head
    println(dataframe1)

    bart.setDoSample(true)

    dataframe1 = model
      .transform(testData)
      .select("summaries.result")
      .collect()
      .toSeq
      .head
      .getAs[Seq[String]](0)
      .head
    println(dataframe1)

  }

}
