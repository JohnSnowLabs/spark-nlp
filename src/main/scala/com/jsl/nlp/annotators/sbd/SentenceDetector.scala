package com.jsl.nlp.annotators.sbd

import com.jsl.nlp.{Annotation, Annotator, Document}
import org.apache.spark.ml.param.Param
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}

/**
  * Created by Saif Addin on 5/5/2017.
  */
class SentenceDetector(override val uid: String) extends Annotator {

  val model: Param[SBDApproach] = new Param(this, "Sentence Detection model", "Approach to detect sentence boundaries")

  override val aType: String = SentenceDetector.aType

  override var requiredAnnotationTypes: Array[String] = Array()

  def this() = this(Identifiable.randomUID(SentenceDetector.aType))

  def getModel: SBDApproach = $(model)

  def setModel(targetModel: SBDApproach): this.type = set(model, targetModel)

  override def annotate(document: Document, annotations: Seq[Annotation]): Seq[Annotation] = {
    val sentences: Seq[Sentence] =
      getModel
        .setContent(document.text)
        .prepare
        .extract
    sentences.map(sentence => Annotation(
      this.aType,
      sentence.begin,
      sentence.end,
      Map[String, String](this.aType -> sentence.content)
    ))
  }

}
object SentenceDetector extends DefaultParamsReadable[SentenceDetector] {
  val aType = "sbd"
}