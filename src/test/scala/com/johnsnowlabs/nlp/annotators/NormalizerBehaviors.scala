package com.johnsnowlabs.nlp.annotators

import com.johnsnowlabs.nlp.{Annotation, AnnotatorBuilder, AnnotatorType}
import org.apache.spark.sql.{Dataset, Row}
import org.scalatest._

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
        val tokens = row.getSeq[Row](3).map(Annotation(_))
        val normalizedAnnotations = row.getSeq[Row](4).map(Annotation(_))
        normalizedAnnotations.foreach {
          case stem: Annotation if stem.annotatorType == AnnotatorType.TOKEN =>
            assert(stem.result.nonEmpty, "Annotation result exists")
          case _ =>
        }

        normalizedAnnotations.zip(tokens).foreach {
          case (stem: Annotation, token: Annotation) =>
            assert(stem.result == token.result.replaceAll("[^a-zA-Z]", ""))
        }
      }
    }
  }
}
