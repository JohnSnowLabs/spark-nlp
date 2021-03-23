package com.johnsnowlabs.nlp

import com.johnsnowlabs.tags.FastTest
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}
import org.scalatest._



class DummyAnnotatorModel(override val uid: String) extends AnnotatorModel[DummyAnnotatorModel] with HasSimpleAnnotate[DummyAnnotatorModel] {
  import AnnotatorType._
  override val outputAnnotatorType: AnnotatorType = DUMMY
  override val inputAnnotatorTypes: Array[AnnotatorType] = Array.empty[AnnotatorType]
  def this() = this(Identifiable.randomUID("DUMMY"))
  setDefault(inputCols, Array.empty[String])
  override def annotate(annotations: Seq[Annotation]): Seq[Annotation] =
    Seq(Annotation(
      outputAnnotatorType,
      0,
      25,
      "dummy result",
      Map("a" -> "b", "c" -> "d")
    ))
}
object DummyAnnotatorModel extends DefaultParamsReadable[DummyAnnotatorModel]

class DemandingDummyAnnotatorModel(override val uid: String) extends AnnotatorModel[DemandingDummyAnnotatorModel] with HasSimpleAnnotate[DemandingDummyAnnotatorModel] {
  import AnnotatorType._
  override val outputAnnotatorType: AnnotatorType = DUMMY
  override val inputAnnotatorTypes: Array[AnnotatorType] = Array(DUMMY)
  def this() = this(Identifiable.randomUID("DEMANDING_DUMMY"))
  setDefault(inputCols, Array(DUMMY))
  override def annotate(annotations: Seq[Annotation]): Seq[Annotation] =
    Seq(Annotation(
      DUMMY,
      11,
      18,
      "dummy result",
      Map("aa" -> "bb", "cc" -> "dd")
    ))
}
object DemandingDummyAnnotatorModel extends DefaultParamsReadable[DemandingDummyAnnotatorModel]
class AnnotatorBaseTestSpec extends FlatSpec {

  val dummyAnnotator = new DummyAnnotatorModel()
    .setOutputCol("dummy")
  val demandingDummyAnnotator = new DemandingDummyAnnotatorModel()
    .setInputCols(Array("dummy"))
    .setOutputCol("demanding_dummy")
  val dummyData = DataBuilder.basicDataBuild("Some dummy content")

  "a dummyAnnotator" should "default inputCols should be an empty one" taggedAs FastTest in {
    assert(dummyAnnotator.getInputCols.isEmpty)
  }

  "a dummyAnnotator" should "have annotation type as an output column" taggedAs FastTest in {
    assert(dummyAnnotator.getOutputCol == "dummy")
  }

  "a demandingDummyAnnotator" should "have input columns as dummy annotator by default" taggedAs FastTest in {
    assert(
      demandingDummyAnnotator.getInputCols.length == 1 &&
        demandingDummyAnnotator.getInputCols.head == "dummy"
    )
  }

  "a demandingDummyAnnotator" should "have annotation type as an output column" taggedAs FastTest in {
    assert(demandingDummyAnnotator.getOutputCol == "demanding_dummy")
  }

  "dummy annotators" should "transform data with default params" taggedAs FastTest in {
    val result = demandingDummyAnnotator.transform(dummyAnnotator.transform(dummyData))
    assert(result.columns.contains(dummyAnnotator.getOutputCol) &&
      result.columns.contains(demandingDummyAnnotator.getOutputCol)
    )
  }

  "dummy annotators" should "transform data with changed params" taggedAs FastTest in {
    dummyAnnotator
      .setOutputCol("demand")
    demandingDummyAnnotator
      .setInputCols(Array("demand"))
      .setOutputCol("result")
    val result = demandingDummyAnnotator.transform(dummyAnnotator.transform(dummyData))
    assert(result.columns.contains("demand") &&
      result.columns.contains("result")
    )
  }

  "dummy annotators" should "transform schema and reflect content metadata as well as schema metadata" taggedAs FastTest in {
    dummyAnnotator
      .setOutputCol("demand")
    demandingDummyAnnotator
      .setInputCols(Array("demand"))
      .setOutputCol("result")
    val result = demandingDummyAnnotator.transform(dummyAnnotator.transform(dummyData))
    val schemaMetadata = result.select("result").schema.fields.head.metadata
    assert(schemaMetadata.contains("annotatorType") &&
      schemaMetadata.getString("annotatorType") == demandingDummyAnnotator.outputAnnotatorType
    )
    import org.apache.spark.sql.Row
    val contentMeta = result.select("demand", "result").take(1).head.getSeq[Row](0)
    val contentAnnotation = contentMeta.map(Annotation(_)).head
    assert(contentAnnotation.annotatorType == dummyAnnotator.outputAnnotatorType)
    assert(contentAnnotation.begin == 0)
    assert(contentAnnotation.end == 25)
    assert(contentAnnotation.metadata.contains("a") && contentAnnotation.metadata("a") == "b")
    val demandContentMeta = result.select("demand", "result").take(1).head.getSeq[Row](1)
    val demandContentAnnotation = demandContentMeta.map(Annotation(_)).head
    assert(demandContentAnnotation.annotatorType == demandingDummyAnnotator.outputAnnotatorType)
    assert(demandContentAnnotation.begin == 11)
    assert(demandContentAnnotation.end == 18)
    assert(demandContentAnnotation.metadata.contains("aa") && demandContentAnnotation.metadata("aa") == "bb")
  }

  "demanding dummy annotator" should "fail if input columns are not found" taggedAs FastTest in {
    dummyAnnotator
      .setOutputCol("demandTypo")
    demandingDummyAnnotator
      .setInputCols(Array("demand"))
      .setOutputCol("result")
    assertThrows[IllegalArgumentException](demandingDummyAnnotator.transform(dummyAnnotator.transform(dummyData)))
  }

}
