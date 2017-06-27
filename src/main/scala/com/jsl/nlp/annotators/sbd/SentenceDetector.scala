package com.jsl.nlp.annotators.sbd

import com.jsl.nlp.annotators.param.AnnotatorParam
import com.jsl.nlp.annotators.sbd.pragmatic.SerializedSBDApproach
import com.jsl.nlp.{Annotation, Annotator, Document}
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}

/**
  * Created by Saif Addin on 5/5/2017.
  */
class SentenceDetector(override val uid: String) extends Annotator {

  val model: AnnotatorParam[SBDApproach, SerializedSBDApproach] =
    new AnnotatorParam[SBDApproach, SerializedSBDApproach](this, "Sentence Detection model", "Approach to detect sentence boundaries")

  override val annotatorType: String = SentenceDetector.aType

  override var requiredAnnotatorTypes: Array[String] = Array()

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
      this.annotatorType,
      sentence.begin,
      sentence.end,
      Map[String, String](this.annotatorType -> sentence.content)
    ))
  }

}
object SentenceDetector extends DefaultParamsReadable[SentenceDetector] {
  val aType = "sbd"
}