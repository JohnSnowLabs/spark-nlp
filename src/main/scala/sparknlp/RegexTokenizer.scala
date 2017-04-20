package sparknlp

import org.apache.spark.ml.param.Param

/**
  * Created by alext on 10/23/16.
  */
class RegexTokenizer() extends Annotator {
  override val aType: String = "token"

  override val requiredAnnotationTypes: Seq[String] = Seq()

  val pattern: Param[String] = new Param(this, "pattern", "this is the token pattern")

  def setPattern(value: String) = set(pattern, value)

  def getPattern: String = $(pattern)

  setDefault(pattern, "\\w+")

  lazy val regex = $(pattern).r

  override def annotate(
    document: Document, annos: Seq[Annotation]
  ): Seq[Annotation] = regex.findAllMatchIn(document.text).map {
    m =>
      Annotation(aType, m.start, m.end)
  }.toSeq
}
