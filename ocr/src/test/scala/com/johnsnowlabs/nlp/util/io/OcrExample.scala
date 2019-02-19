package com.johnsnowlabs.nlp.util.io

import java.io.File

import com.johnsnowlabs.nlp.{DocumentAssembler, LightPipeline}
import com.johnsnowlabs.util.OcrMetrics
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.SparkSession
import org.scalatest._
import javax.imageio.ImageIO
import scala.io.Source


class OcrExample extends FlatSpec with ImageProcessing with OcrMetrics {

  trait Scope extends ImageProcessing

  "OcrHelper" should "correctly analyze window" in new Scope {

    val window1 = Array(1, 1, 1, 1, 5, 5, 5, 1, 1, 1, 1)
    assert(analyzeWindow(window1)._2 == 1)

    val window2 = Array(1, 1, 1, 5, 5, 5, 5, 5, 1, 1, 1)
    assert(analyzeWindow(window2)._2 == 2)

    val window3 = Array(1, 1, 1, 1, 5, 5, 1, 1, 1, 1)
    assert(analyzeWindow(window3)._2 == 1)

    val window4 = Array(1, 1, 1, 5, 5, 5, 5, 1, 1, 1)
    assert(analyzeWindow(window4)._2 == 2)
  }

  "OcrHelper" should "correctly detect font size" in {
    val img1 = ImageIO.read(new File("ocr/src/test/resources/images/big_font.jpg"))
    val bigSize = detectFontSize(convertToGrayScale(img1)).get
    assert(bigSize > 70 && bigSize < 90)

    val img2 = ImageIO.read(new File("ocr/src/test/resources/images/small_font.jpg"))
    val smallSize = detectFontSize(convertToGrayScale(img2)).get
    assert(smallSize > 35 && smallSize < 45)
  }

  "OcrHelper" should "improve accuracy with automatic scaling" in {
    val spark = getSpark

    val small = OcrHelper.createDataset(spark, s"ocr/src/test/resources/pdfs/scaled/toosmall").
      select("text").collect.map(_.getString(0)).mkString

    OcrHelper.setAutomaticSizeCorrection(true)
    val sizeCorrected = OcrHelper.createDataset(spark, s"ocr/src/test/resources/pdfs/scaled/toosmall").
      select("text").collect.map(_.getString(0)).mkString

    val correct = Source.fromFile("ocr/src/test/resources/pdfs/scaled/correct.txt").mkString
    assert(score(correct, small) < score(correct, sizeCorrected))
  }

  "Sign convertions" should "map all the values back and forwards" in {
    (-128 to 127).map(_.toByte).foreach { b=>
      assert(b == unsignedInt2signedByte(signedByte2UnsignedInt(b)))
    }
  }

  "OcrHelper" should "correctly threshold and invert images" in {
    val img = ImageIO.read(new File("ocr/src/test/resources/images/p1.jpg"))
    val tresImg = thresholdAndInvert(img, 205, 255)
    dumpImage(tresImg, "thresholded_binarized.png")
  }

  "OcrHelper" should "correctly detect and correct skew angles" in {
    val img = ImageIO.read(new File("ocr/src/test/resources/images/p1.jpg"))
    val correctedImg = correctSkew(img, 2.0, 1.0)
    dumpImage(correctedImg, "skew_corrected.png")
  }

 "OcrHelper" should "automatically correct skew and improve accuracy" in {
    OcrHelper.setAutomaticSizeCorrection(false)
    val spark = getSpark
    val normal = OcrHelper.createDataset(spark, s"ocr/src/test/resources/pdfs/rotated/400").
       select("text").collect.map(_.getString(0)).mkString

    OcrHelper.setAutomaticSkewCorrection(true)
    val skewCorrected = OcrHelper.createDataset(spark, s"ocr/src/test/resources/pdfs/rotated/400").
        select("text").collect.map(_.getString(0)).mkString

    val correct = Source.fromFile("ocr/src/test/resources/pdfs/rotated/400.txt").mkString
    assert(score(correct, normal) < score(correct, skewCorrected))
  } 

  "OcrHelper" should "correctly handle PDFs with multiple images" in {

    val spark = getSpark
    OcrHelper.setPreferredMethod(OCRMethod.IMAGE_LAYER)
    OcrHelper.setSplitPages(false)

    val multiple = OcrHelper.createDataset(spark, "ocr/src/test/resources/pdfs/multiple").
      select("text").collect.map(_.getString(0)).mkString

    val single = OcrHelper.createDataset(spark, "ocr/src/test/resources/pdfs/single").
      select("text").collect.map(_.getString(0)).mkString

    assert(levenshteinDistance(multiple, single) < 100)

  }

  "OcrExample with Spark" should "successfully create a dataset" in {

      val spark = getSpark
      import spark.implicits._

      // point to test/resources/pdfs
      val data = OcrHelper.createDataset(spark, "ocr/src/test/resources/pdfs/multiple")
      OcrHelper.setPreferredMethod(OCRMethod.IMAGE_LAYER)
      data.show(10)
      val documentAssembler = new DocumentAssembler().setInputCol("text")
      documentAssembler.transform(data).show()
      val raw = OcrHelper.createMap("ocr/src/test/resources/pdfs/")
      val pipeline = new LightPipeline(new Pipeline().setStages(Array(documentAssembler)).fit(Seq.empty[String].toDF("text")))
      val result = pipeline.annotate(raw.values.toArray)

      assert(raw.size == 2 && result.nonEmpty)
      println(result.mkString(","))
      succeed
  }

  "OcrExample with Spark" should "improve results when preprocessing images" in {
      val spark = getSpark
      OcrHelper.setScalingFactor(3.0f)
      OcrHelper.useErosion(true, kSize = 2)
      val data = OcrHelper.createDataset(spark, "ocr/src/test/resources/pdfs/problematic")
      val results = data.select("text").collect.flatMap(_.getString(0).split("\n")).toSet
      assert(results.contains("1.5"))
      assert(results.contains("223.5"))
      assert(results.contains("22.5"))

  }

  def getSpark = {
        SparkSession.builder()
          .appName("SparkNLP-OCR-Default-Spark")
          .master("local[*]")
          .config("spark.driver.memory", "4G")
          .config("spark.driver.maxResultSize", "2G")
          .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
          .config("spark.kryoserializer.buffer.max", "500m")
          .getOrCreate()
    }
}
