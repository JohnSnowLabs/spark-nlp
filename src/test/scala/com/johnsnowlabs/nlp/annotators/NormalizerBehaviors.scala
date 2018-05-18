package com.johnsnowlabs.nlp.annotators

import com.johnsnowlabs.nlp.util.io.{ExternalResource, ReadAs}
import com.johnsnowlabs.nlp._
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.{Dataset, Row}
import org.scalatest._
import SparkAccessor.spark.implicits._
import com.johnsnowlabs.nlp.pretrained.pipelines.en.BasicPipeline

trait NormalizerBehaviors { this: FlatSpec =>

  def fullNormalizerPipeline(dataset: => Dataset[Row]) {
    "A Normalizer Annotator" should "successfully transform data" in {
    AnnotatorBuilder.withFullNormalizer(dataset)
      .collect().foreach {
      row =>
        row.getSeq[Row](4)
          .map(Annotation(_))
          .foreach {
            case stem: Annotation if stem.annotatorType == AnnotatorType.TOKEN =>
              assert(stem.result.nonEmpty, "Annotation result exists")
            case _ =>
          }
      }
    }
  }

  def lowercasingNormalizerPipeline(dataset: => Dataset[Row]) {
    "A case-sensitive Normalizer Annotator" should "successfully transform data" in {
    AnnotatorBuilder.withCaseSensitiveNormalizer(dataset)
      .collect().foreach {
      row =>
        val tokens = row.getSeq[Row](3).map(Annotation(_)).filterNot(a => a.result == "." || a.result == ",")
        val normalizedAnnotations = row.getSeq[Row](4).map(Annotation(_))
        normalizedAnnotations.foreach {
          case nToken: Annotation if nToken.annotatorType == AnnotatorType.TOKEN =>
            assert(nToken.result.nonEmpty, "Annotation result exists")
          case _ =>
        }
        normalizedAnnotations.zip(tokens).foreach {
          case (nToken: Annotation, token: Annotation) =>
            assert(nToken.result == token.result.replaceAll("[^a-zA-Z]", ""))
        }
      }
    }
  }

  def testCorrectSlangs(dataset: Dataset[Row]): Unit = {
    s"normalizer with slang dictionary " should s"successfully correct several abbreviations" in {

      val documentAssembler = new DocumentAssembler()
        .setInputCol("text")
        .setOutputCol("document")

      val tokenizer = new Tokenizer()
        .setInputCols(Array("document"))
        .setOutputCol("token")

      val normalizer = new Normalizer()
        .setInputCols(Array("token"))
        .setOutputCol("normalized")
        .setLowercase(true)
        .setSlangDictionary(ExternalResource("src/test/resources/spell/slangs.txt",
                            ReadAs.LINE_BY_LINE, Map("delimiter" -> ",")))

      val finisher = new Finisher()
        .setInputCols("normalized")
        .setOutputAsArray(false)
        .setIncludeKeys(false)

      val pipeline = new Pipeline()
        .setStages(Array(
          documentAssembler,
          tokenizer,
          normalizer,
          finisher
        ))

      val model = pipeline.fit(DataBuilder.basicDataBuild("dummy"))
      val transform = model.transform(dataset)
      transform.show()
      val normalizedWords = transform.select("normalized_gt",
        "finished_normalized").map(r => (r.getString(0), r.getString(1))).collect.toSeq

      normalizedWords.foreach( words => {
        assert(words._1 == words._2)
      })
    }
  }

  def testMultipleRegexPatterns(dataset: Dataset[Row]): Unit = {
    s"normalizer with multiple regex patterns " should s"successfully correct several words" in {

      val documentAssembler = new DocumentAssembler()
        .setInputCol("text")
        .setOutputCol("document")

      val tokenizer = new Tokenizer()
        .setInputCols(Array("document"))
        .setOutputCol("token")

      val normalizer = new Normalizer()
        .setInputCols(Array("token"))
        .setOutputCol("normalized")
        .setLowercase(false)
        .setPattern(Array("[^\\pL+]", "[^a-z]"))

      val finisher = new Finisher()
        .setInputCols("normalized")
        .setOutputAsArray(false)
        .setIncludeKeys(false)

      val pipeline = new Pipeline()
        .setStages(Array(
          documentAssembler,
          tokenizer,
          normalizer,
          finisher
        ))

      val model = pipeline.fit(DataBuilder.basicDataBuild("dummy"))
      val transform = model.transform(dataset)
      transform.show()
      val normalizedWords = transform.select("normalized_gt",
        "finished_normalized").map(r => (r.getString(0), r.getString(1))).collect.toSeq

      normalizedWords.foreach( words => {
        assert(words._1 == words._2)
      })
    }
  }

  def testLoadModel(): Unit = {
    s"a Normalizer annotator with a load model" should
      "successfully normalize words" in {
      val data = Seq("gr8").toDS.toDF("text")
      data.show()
      val path = "./tmp_normalizer"

      val documentAssembler = new DocumentAssembler()
        .setInputCol("text")
        .setOutputCol("document")

      val tokenizer = new Tokenizer()
        .setInputCols(Array("document"))
        .setOutputCol("token")

      val normalizer = new Normalizer()
        .setInputCols(Array("token"))
        .setOutputCol("normal")
        .setSlangDictionary("src/test/resources/spell/slangs.txt",",")

      val trainNormalizer = normalizer.fit(data)
      trainNormalizer.write.overwrite.save(path)

      val loadedNormalizer = Normalizer.load(path)

      val finisher = new Finisher()
        .setInputCols("normal")
        .setOutputAsArray(false)
        .setIncludeKeys(false)

      val pipeline = new Pipeline()
        .setStages(Array(
          documentAssembler,
          tokenizer,
          loadedNormalizer,
          finisher
        ))

      val model = pipeline.fit(DataBuilder.basicDataBuild("dummy"))
      val transform = model.transform(data)
      transform.show()
      val normalizedWords = transform.select("finished_normal").map(r => r.getString(0)).collect.toList
      assert("great" == normalizedWords.head)
    }
  }

}
