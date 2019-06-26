package com.johnsnowlabs.nlp.util.io

import java.io.File

import com.johnsnowlabs.nlp.annotators.spell.util.Utilities
import com.johnsnowlabs.nlp.{DocumentAssembler, LightPipeline}
import com.johnsnowlabs.util.OcrMetrics
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.SparkSession
import org.scalatest._
import javax.imageio.ImageIO

import scala.io.Source


class OcrExample extends FlatSpec with ImageProcessing with OcrMetrics {

  val ocrHelper = new OcrHelper()

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

   "OcrHelper" should "correctly compute noise measures for input images" in {
    import org.apache.spark.sql.functions._
    ocrHelper.setEstimateNoise(NoiseMethod.VARIANCE)
    ocrHelper.setPreferredMethod(OCRMethod.IMAGE_LAYER)

    val noisyScore = ocrHelper.createDataset(getSpark, s"ocr/src/test/resources/noisy_images").
      select(avg("noiselevel").as("average")).collect().
      map(_.getAs[Double]("average")).head

    val cleanScore = ocrHelper.createDataset(getSpark, s"ocr/src/test/resources/noiseless_images").
      select(avg("noiselevel").as("average")).collect().
      map(_.getAs[Double]("average")).head

    assert(noisyScore > cleanScore)
  }

  "OcrHelper" should "correctly return confidence levels for recognized text" in {
    import org.apache.spark.sql.functions._
    ocrHelper.setIncludeConfidence(true)

    val noisyScore = ocrHelper.createDataset(getSpark, s"ocr/src/test/resources/noisy_images").
      select(avg("confidence").as("average")).collect().
      map(_.getAs[Double]("average")).head

    val cleanScore = ocrHelper.createDataset(getSpark, s"ocr/src/test/resources/noiseless_images").
      select(avg("confidence").as("average")).collect().
      map(_.getAs[Double]("average")).head

    assert(noisyScore < cleanScore)
  }

 "OcrHelper" should "automatically correct skew and improve accuracy" in {
    val spark = getSpark
    ocrHelper.setPreferredMethod(OCRMethod.IMAGE_LAYER)
    ocrHelper.setSplitPages(false)

    val normal = ocrHelper.createDataset(spark, s"ocr/src/test/resources/pdfs/rotated/400").
       select("text").collect.map(_.getString(0)).mkString

    ocrHelper.setAutomaticSkewCorrection(true)

    val skewCorrected = ocrHelper.createDataset(spark, s"ocr/src/test/resources/pdfs/rotated/400").
        select("text").collect.map(_.getString(0)).mkString

    val correct = Source.fromFile("ocr/src/test/resources/pdfs/rotated/400.txt").mkString
    assert(score(correct, normal) < score(correct, skewCorrected))
  }

  "OcrHelper" should "correctly handle PDFs with multiple images" in {

    val spark = getSpark
    ocrHelper.setPreferredMethod(OCRMethod.IMAGE_LAYER)
    ocrHelper.setSplitPages(true)

    val multiple = ocrHelper.createDataset(spark, "ocr/src/test/resources/pdfs/multiple").
      select("text").collect.map(_.getString(0)).mkString

    val single = ocrHelper.createDataset(spark, "ocr/src/test/resources/pdfs/single").
      select("text").collect.map(_.getString(0)).mkString

    assert(Utilities.levenshteinDistance(multiple, single) < 102)

  }

  "OcrHelper" should "correctly count pages" in {
    val spark = getSpark
    import spark.implicits._

    ocrHelper.setFallbackMethod(true)
    ocrHelper.setPreferredMethod(OCRMethod.IMAGE_LAYER)
    ocrHelper.setScalingFactor(2.0f)
    val data = ocrHelper.createDataset(getSpark, "ocr/src/test/resources/pdfs/multiplepages")
    //got 6 pages
    assert(data.select("filename", "pagenum").count == 6)
    //got 2 files
    assert(data.select("filename").distinct.count == 2)
    val filePage = data.select("filename", "pagenum").as[(String, Int)].collect
    assert(filePage.count(_._2 == 0) == 2)
    assert(filePage.count(_._2 == 1) == 2)
    assert(filePage.count(_._2 == 2) == 2)

    ocrHelper.setSplitPages(false)
    ocrHelper.setMinSizeBeforeFallback(10)
    val raw = ocrHelper.createMap("ocr/src/test/resources/pdfs/multiplepages")
    assert(raw.size == 2)
    // check content from three pages is present on each doc
    assert(raw.forall(_._2.size > 50))

  }

  "OcrExample with Spark" should "successfully create a dataset - PDFs" in {

      val spark = getSpark
      import spark.implicits._

      // point to test/resources/pdfs
      ocrHelper.setSplitRegions(false)
      ocrHelper.setFallbackMethod(true)
      ocrHelper.setMinSizeBeforeFallback(10)

      val data = ocrHelper.createDataset(spark, "ocr/src/test/resources/pdfs")
      val documentAssembler = new DocumentAssembler().setInputCol("text")
      documentAssembler.transform(data).show()

      val raw = ocrHelper.createMap("ocr/src/test/resources/pdfs/")
      val pipeline = new LightPipeline(new Pipeline().setStages(Array(documentAssembler)).fit(Seq.empty[String].toDF("text")))
      val result = pipeline.annotate(raw.values.toArray)

      assert(raw.size == 2 && result.nonEmpty)
      println(result.mkString(","))
      succeed
  }

  "OcrExample with Spark" should "successfully create a dataset - images" in {

    val spark = getSpark
    import spark.implicits._
    ocrHelper.setSplitPages(false)

    val data = ocrHelper.createDataset(spark, "ocr/src/test/resources/images/").
      select("text").collect().mkString(" ")

    val correct = Source.fromFile("ocr/src/test/resources/txt/p1.txt").mkString
    assert(Utilities.levenshteinDistance(correct, data) < 900)
  }


  "OcrExample with Spark" should "improve results when preprocessing images" in {
      val spark = getSpark
      ocrHelper.setScalingFactor(3.0f)
      ocrHelper.useErosion(true, kSize = 2)
      val data = ocrHelper.createDataset(spark, "ocr/src/test/resources/pdfs/problematic")
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
