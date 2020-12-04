package com.johnsnowlabs.nlp.annotators.ld.dl

import com.johnsnowlabs.nlp.annotators.sbd.pragmatic.SentenceDetector
import com.johnsnowlabs.nlp.annotators.sentence_detector_dl.SentenceDetectorDLModel
import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.functions.{input_file_name, monotonically_increasing_id, regexp_extract, sum, udf, when}
import org.scalatest._

class LanguageDetectorDLTestSpec extends FlatSpec {

  val spark = ResourceHelper.spark

  "LanguageDetectorDL" should "correctly load pretrained model" in {

    val smallCorpus = spark.read
      .option("header", true)
      .option("delimiter", "|")
      .csv("src/test/resources/language-detector/multilingual_sample.txt")

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val sentence = SentenceDetectorDLModel.pretrained()
      .setInputCols(Array("document"))
      .setOutputCol("sentence")

    val languageDetector = LanguageDetectorDL.pretrained()
      .setInputCols("sentence")
      .setOutputCol("language")

    val pipeline = new Pipeline()
      .setStages(Array(
        documentAssembler,
        sentence,
        languageDetector
      ))

    val pipelineDF = pipeline.fit(smallCorpus).transform(smallCorpus)
    println(pipelineDF.count())
    smallCorpus.show(2)
    pipelineDF.show(2)
    pipelineDF.select("sentence").show(4, false)
    pipelineDF.select("language.metadata").show(20, false)
    pipelineDF.select("language.result", "lang").show(20, false)
    pipeline.fit(smallCorpus).write.overwrite().save("./tmp_ld_pipeline")
    val pipelineModel = PipelineModel.load("./tmp_ld_pipeline")
    pipelineModel.transform(smallCorpus).select("language.result", "lang").show(20, false)

  }

}
