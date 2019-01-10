package com.johnsnowlabs.example

import java.io.{File, PrintWriter}

import com.johnsnowlabs.nlp.DocumentAssembler
import com.johnsnowlabs.nlp.annotators.Tokenizer
import com.johnsnowlabs.nlp.util.io.OcrHelper
import com.johnsnowlabs.nlp.SparkAccessor.spark
import net.sourceforge.tess4j.ITessAPI.TessPageSegMode
import org.apache.spark.ml.Pipeline

object SampleImageOCR extends App {

  import spark.implicits._

  val srcPath = "/home/jose/Downloads/ocr_evaluation/test_set_images"
  val dstPath = "/home/jose/Downloads/ocr_evaluation/sparknlp_output"

  OcrHelper.setScalingFactor(1.0f)
  OcrHelper.useErosion(false, 1)
  OcrHelper.setPageSegMode(TessPageSegMode.PSM_SINGLE_BLOCK)
  val data = OcrHelper.createDataset(spark,srcPath).cache

  val documentAssembler = new DocumentAssembler().
    setTrimAndClearNewLines(false).
    setInputCol("text").
    setOutputCol("document")

  val tokenizer: Tokenizer = new Tokenizer()
    .setInputCols(Array("document"))
    .setOutputCol("token")

  val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer))
  val fitPipeline = pipeline.fit(data)
  val fnames = fitPipeline.transform(data).select("filename").as[String]
    .collect().sorted.map(_.split("/").takeRight(1).head).toSet

  val allFiles = new File(srcPath).listFiles.filter(_.isFile).map(_.getName).toSet

  val diff = allFiles -- fnames
  val recognizedTexts = fitPipeline.transform(data)
  val texts = fitPipeline.transform(data).select("text").as[String].collect
  val names = fitPipeline.transform(data).select("filename").as[String].collect.
    map(_.split("/").takeRight(1).head).map(name => s"""$name""").toList

  texts.zip(names).foreach { case (text, name) =>
    val file = new File(s"""$dstPath/$name.txt""")
    val pw = new PrintWriter(file)
    pw.write(text)
    pw.close
  }

  print(names.map(name => s"""\"$name\""""))
}
