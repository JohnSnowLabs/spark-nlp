package com.johnsnowlabs.nlp.annotators.ner
import com.johnsnowlabs.nlp.{Annotation, AnnotatorModel, HasSimpleAnnotate}
import org.apache.spark.ml.param.{Param, StringArrayParam}
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}

/**
  * @groupname anno Annotator types
  * @groupdesc anno Required input and expected output annotator types
  * @groupname Ungrouped Members
  * @groupname param Parameters
  * @groupname setParam Parameter setters
  * @groupname getParam Parameter getters
  * @groupname Ungrouped Members
  * @groupprio param  1
  * @groupprio anno  2
  * @groupprio Ungrouped 3
  * @groupprio setParam  4
  * @groupprio getParam  5
  * @groupdesc param A list of (hyper-)parameter keys this annotator can take. Users can set and get the parameter values through setters and getters, respectively.
  */
class NerOverwriter(override val uid: String) extends AnnotatorModel[NerOverwriter] with HasSimpleAnnotate[NerOverwriter] {

  import com.johnsnowlabs.nlp.AnnotatorType.NAMED_ENTITY

  /** Output Annotator Type : NAMED_ENTITY
    *
    * @group anno
    **/
  override val outputAnnotatorType: AnnotatorType = NAMED_ENTITY
  /** Input Annotator Type : NAMED_ENTITY
    *
    * @group anno
    **/
  override val inputAnnotatorTypes: Array[AnnotatorType] = Array(NAMED_ENTITY)

  def this() = this(Identifiable.randomUID("NER_OVERWRITER"))

  /** the words to be filtered out.
    *
    * @group param
    **/
  val stopWords: StringArrayParam = new StringArrayParam(this, "stopWords", "the words to be filtered out.")

  /** the words to be filtered out.
    *
    * @group setParam
    **/
  def setStopWords(value: Array[String]): this.type = set(stopWords, value)

  /** the words to be filtered out.
    *
    * @group getParam
    **/
  def getStopWords: Array[String] = $(stopWords)

  /** New NER class to overwrite
    *
    * @group param
    **/
  val newResult: Param[String] = new Param(this, "newResult", "New NER class to overwrite")

  /** New NER class to overwrite
    *
    * @group setParam
    **/
  def setNewResult(r: String): this.type = {
    set(newResult, r)
  }

  /** New NER class to overwrite
    *
    * @group getParam
    **/
  def getNewResult: String = $(newResult)

  setDefault(
    newResult -> "I-OVERWRITE"
  )

  override def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {

    var annotationsOverwritten = annotations

    annotationsOverwritten.map { tokenAnnotation =>
      val stopWordsSet = $(stopWords).toSet
      if (stopWordsSet.contains(tokenAnnotation.metadata("word"))) {
        Annotation(
          outputAnnotatorType,
          tokenAnnotation.begin,
          tokenAnnotation.end,
          $(newResult),
          tokenAnnotation.metadata
        )
      } else {
        Annotation(
          outputAnnotatorType,
          tokenAnnotation.begin,
          tokenAnnotation.end,
          tokenAnnotation.result,
          tokenAnnotation.metadata
        )
      }

    }

  }

}

object NerOverwriter extends DefaultParamsReadable[NerOverwriter]
