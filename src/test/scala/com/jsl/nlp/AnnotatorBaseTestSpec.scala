package com.jsl.nlp

import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}
import org.apache.spark.sql.AnalysisException
import org.scalatest._

/**
  * Created by Saif Addin on 6/1/2017.
  */
class AnnotatorBaseTestSpec extends FlatSpec {

  class DummyAnnotator(override val uid: String) extends Annotator {
    override val annotatorType: String = DummyAnnotator.aType
    override var requiredAnnotatorTypes: Array[String] = Array()
    def this() = this(Identifiable.randomUID(DummyAnnotator.aType))
    override def annotate(document: Document, annotations: Seq[Annotation]): Seq[Annotation] =
      Seq(Annotation(
        DummyAnnotator.aType,
        0,
        25,
        Map("a" -> "b", "c" -> "d")
      ))
  }
  object DummyAnnotator extends DefaultParamsReadable[DummyAnnotator] {
    val aType = "dummyType"
  }

  class DemandingDummyAnnotator(override val uid: String) extends Annotator {
    override val annotatorType: String = DemandingDummyAnnotator.aType
    override var requiredAnnotatorTypes: Array[String] = Array(DummyAnnotator.aType)
    def this() = this(Identifiable.randomUID(DemandingDummyAnnotator.aType))
    override def annotate(document: Document, annotations: Seq[Annotation]): Seq[Annotation] =
      Seq(Annotation(
        DemandingDummyAnnotator.aType,
        11,
        18,
        Map("aa" -> "bb", "cc" -> "dd")
      ))
  }
  object DemandingDummyAnnotator extends DefaultParamsReadable[DemandingDummyAnnotator] {
    val aType = "demandingDummyType"
  }

  val dummyAnnotator = new DummyAnnotator
  val demandingDummyAnnotator = new DemandingDummyAnnotator
  val dummyData = DataBuilder.basicDataBuild("Some dummy content")

  "a dummyAnnotator" should "not have any input columns set by default" in {
    assert(dummyAnnotator.getInputAnnotationCols.isEmpty)
  }

  "a dummyAnnotator" should "have annotation type as an output column" in {
    assert(dummyAnnotator.getOutputAnnotationCol == DummyAnnotator.aType)
  }

  "a demandingDummyAnnotator" should "have input columns as dummy annotator by default" in {
    assert(
      demandingDummyAnnotator.getInputAnnotationCols.length == 1 &&
        demandingDummyAnnotator.getInputAnnotationCols.head == DummyAnnotator.aType
    )
  }

  "a demandingDummyAnnotator" should "have annotation type as an output column" in {
    assert(demandingDummyAnnotator.getOutputAnnotationCol == DemandingDummyAnnotator.aType)
  }

  "a dummyAnnotator" should "not have any document column set and return a proper one after set" in {
    assertThrows[NoSuchElementException](dummyAnnotator.getDocumentCol)
    dummyAnnotator.setDocumentCol("document")
    assert(dummyAnnotator.getDocumentCol == "document")
  }

  "dummy annotators" should "transform data with default params" in {
    dummyAnnotator.setDocumentCol("document")
    demandingDummyAnnotator.setDocumentCol("document")
    val result = demandingDummyAnnotator.transform(dummyAnnotator.transform(dummyData))
    assert(result.columns.contains(dummyAnnotator.getOutputAnnotationCol) &&
      result.columns.contains(demandingDummyAnnotator.getOutputAnnotationCol)
    )
  }

  "dummy annotators" should "transform data with changed params" in {
    dummyAnnotator
      .setDocumentCol("document")
      .setOutputAnnotationCol("demand")
    demandingDummyAnnotator
      .setDocumentCol("document")
      .setInputAnnotationCols(Array("demand"))
      .setOutputAnnotationCol("result")
    val result = demandingDummyAnnotator.transform(dummyAnnotator.transform(dummyData))
    assert(result.columns.contains("demand") &&
      result.columns.contains("result")
    )
  }

  "dummy annotators" should "transform schema and reflect content metadata as well as schema metadata" in {
    dummyAnnotator
      .setDocumentCol("document")
      .setOutputAnnotationCol("demand")
    demandingDummyAnnotator
      .setDocumentCol("document")
      .setInputAnnotationCols(Array("demand"))
      .setOutputAnnotationCol("result")
    val result = demandingDummyAnnotator.transform(dummyAnnotator.transform(dummyData))
    val schemaMetadata = result.select("result").schema.fields.head.metadata
    assert(schemaMetadata.contains("annotationType") &&
      schemaMetadata.getString("annotationType") == demandingDummyAnnotator.annotatorType
    )
    import org.apache.spark.sql.Row
    val contentMeta = result.select("demand", "result").collect.head.getSeq[Row](0)
    val contentAnnotation = contentMeta.map(Annotation(_)).head
    assert(contentAnnotation.annotatorType == dummyAnnotator.annotatorType)
    assert(contentAnnotation.begin == 0)
    assert(contentAnnotation.end == 25)
    assert(contentAnnotation.metadata.contains("a") && contentAnnotation.metadata("a") == "b")
    val demandContentMeta = result.select("demand", "result").collect.head.getSeq[Row](1)
    val demandContentAnnotation = demandContentMeta.map(Annotation(_)).head
    assert(demandContentAnnotation.annotatorType == demandingDummyAnnotator.annotatorType)
    assert(demandContentAnnotation.begin == 11)
    assert(demandContentAnnotation.end == 18)
    assert(demandContentAnnotation.metadata.contains("aa") && demandContentAnnotation.metadata("aa") == "bb")
  }

  "demanding dummy annotator" should "fail if input columns are not found" in {
    dummyAnnotator
      .setDocumentCol("document")
      .setOutputAnnotationCol("demandTypo")
    demandingDummyAnnotator
      .setDocumentCol("document")
      .setInputAnnotationCols(Array("demand"))
      .setOutputAnnotationCol("result")
    assertThrows[AnalysisException](demandingDummyAnnotator.transform(dummyAnnotator.transform(dummyData)))
  }

}
