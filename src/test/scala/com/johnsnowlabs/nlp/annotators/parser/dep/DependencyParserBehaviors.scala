package com.johnsnowlabs.nlp.annotators.parser.dep

import com.johnsnowlabs.nlp._
import org.apache.spark.sql.{DataFrame, Dataset, Row}
import org.scalatest.FlatSpec
import com.johnsnowlabs.util.PipelineModels
import org.apache.spark.ml.Pipeline

trait DependencyParserBehaviors { this: FlatSpec =>


  def initialAnnotations(testDataSet: Dataset[Row]): Unit = {
    val fixture = createFixture(testDataSet)
    it should "add annotations" in {
      assert(fixture.dependencies.count > 0, "Annotations count should be greater than 0")
    }

    it should "add annotations with the correct annotationType" in {
      fixture.depAnnotations.foreach { a =>
        assert(a.annotatorType == AnnotatorType.DEPENDENCY, s"Annotation type should ${AnnotatorType.DEPENDENCY}")
      }
    }

    it should "annotate each token" in {
      assert(fixture.tokenAnnotations.size == fixture.depAnnotations.size, s"Every token should be annotated")
    }

    it should "annotate each word with a head" in {
      fixture.depAnnotations.foreach { a =>
        assert(a.result.nonEmpty, s"Result should have a head")
      }
    }

    it should "annotate each word with the correct indexes" in {
      fixture.depAnnotations
        .zip(fixture.tokenAnnotations)
        .foreach { case (dep, token) => assert(dep.begin == token.begin && dep.end == token.end, s"Token and word should have equal indixes") }
    }
  }

  private def createFixture(testDataSet: Dataset[Row]) = new {
    val dependencies: DataFrame = testDataSet.select("dependency")
    val depAnnotations: Seq[Annotation] = dependencies
      .collect
      .flatMap { r => r.getSeq[Row](0) }
      .map { r =>
        Annotation(r.getString(0), r.getInt(1), r.getInt(2), r.getString(3), r.getMap[String, String](4))
      }
    val tokens: DataFrame = testDataSet.select("token")
    val tokenAnnotations: Seq[Annotation] = tokens
      .collect
      .flatMap { r => r.getSeq[Row](0) }
      .map { r =>
        Annotation(r.getString(0), r.getInt(1), r.getInt(2), r.getString(3), r.getMap[String, String](4))
      }
  }

  def relationshipsBetweenWordsPredictor(testDataSet: Dataset[Row], pipeline: Pipeline): Unit = {

    val emptyDataSet = PipelineModels.dummyDataset

    val dependencyParserModel = pipeline.fit(emptyDataSet)

    it should "train a model" in {
      val model = dependencyParserModel.stages.last.asInstanceOf[DependencyParserModel]
      assert(model.isInstanceOf[DependencyParserModel])
    }

    val dependencyParserDataFrame = dependencyParserModel.transform(testDataSet)
    //dependencyParserDataFrame.collect()
    dependencyParserDataFrame.show(false)

    it should "predict relationships between words" in {
      assert(dependencyParserDataFrame.isInstanceOf[DataFrame])
    }

  }

}
