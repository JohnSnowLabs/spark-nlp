package com.johnsnowlabs.nlp.annotators

import com.johnsnowlabs.collections.SearchTrie
import com.johnsnowlabs.nlp.AnnotatorType._
import com.johnsnowlabs.nlp._
import com.johnsnowlabs.nlp.serialization.StructFeature
import org.apache.spark.ml.param.{BooleanParam, Param}
import org.apache.spark.ml.util.Identifiable

import scala.annotation.{tailrec => tco}
import scala.collection.mutable.ArrayBuffer

/**
  * Extracts entities out of provided phrases
  *
  * @param uid internally renquired UID to make it writable
  * @@ entitiesPath: Path to file with phrases to search
  * @@ insideSentences: Should Extractor search only within sentence borders?
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
class TextMatcherModel(override val uid: String) extends AnnotatorModel[TextMatcherModel] with HasSimpleAnnotate[TextMatcherModel] {

  /** Output annotator type : CHUNK
    *
    * @group anno
    **/
  override val outputAnnotatorType: AnnotatorType = CHUNK
  /** input annotator type : DOCUMENT, TOKEN
    *
    * @group anno
    **/
  override val inputAnnotatorTypes: Array[AnnotatorType] = Array(DOCUMENT, TOKEN)

  /** searchTrie for Searching words
    *
    * @group param
    **/
  val searchTrie = new StructFeature[SearchTrie](this, "searchTrie")
  /** whether to merge overlapping matched chunks. Defaults false
    *
    * @group param
    **/
  val mergeOverlapping = new BooleanParam(this, "mergeOverlapping", "whether to merge overlapping matched chunks. Defaults false")
  /** Value for the entity metadata field
    *
    * @group param
    **/
  val entityValue = new Param[String](this, "entityValue", "Value for the entity metadata field")
  /** Whether the TextMatcher should take the CHUNK from TOKEN or not
    *
    * @group param
    **/
  val buildFromTokens = new BooleanParam(this, "buildFromTokens", "Whether the TextMatcher should take the CHUNK from TOKEN or not")

  /** SearchTrie of Tokens
    *
    * @group setParam
    **/
  def setSearchTrie(value: SearchTrie): this.type = set(searchTrie, value)

  /** Whether to merge overlapping matched chunks. Defaults false
    *
    * @group setParam
    **/
  def setMergeOverlapping(v: Boolean): this.type = set(mergeOverlapping, v)

  /** Whether to merge overlapping matched chunks. Defaults false
    *
    * @group getParam
    **/
  def getMergeOverlapping: Boolean = $(mergeOverlapping)

  /** Setter for Value for the entity metadata field
    *
    * @group setParam
    **/
  def setEntityValue(v: String): this.type = set(entityValue, v)

  /** Getter for Value for the entity metadata field
    *
    * @group getParam
    **/
  def getEntityValue: String = $(entityValue)

  /** internal constructor for writabale annotator */
  def this() = this(Identifiable.randomUID("ENTITY_EXTRACTOR"))

  /** Setter for buildFromTokens param
    *
    * @group setParam
    **/
  def setBuildFromTokens(v: Boolean): this.type = set(buildFromTokens, v)

  /** Getter for buildFromTokens param
    *
    * @group getParam
    **/
  def getBuildFromTokens: Boolean = $(buildFromTokens)

  setDefault(inputCols, Array(TOKEN))
  setDefault(mergeOverlapping, false)
  setDefault(entityValue, "entity")

  @tco final protected def collapse(rs: List[(Int, Int)], sep: List[(Int, Int)] = Nil): List[(Int, Int)] = rs match {
    case x :: y :: rest =>
      if (y._1 > x._2) collapse(y :: rest, x :: sep)
      else collapse((x._1, x._2 max y._2) :: rest, sep)
    case _ =>
      (rs ::: sep).reverse
  }
  protected def merge(rs: List[(Int,Int)]): List[(Int,Int)] = collapse(rs.sortBy(_._1))

  /**
    *
    * Searches entities and stores them in the annotation.  Defines annotator phrase matching depending on whether we are using SBD or not
    *
    * @return Extracted Entities
    */
  override def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {

    val result = ArrayBuffer[Annotation]()

    val sentences = annotations.filter(_.annotatorType == AnnotatorType.DOCUMENT)

    sentences.zipWithIndex.foreach{case (sentence, sentenceIndex) =>

      val tokens = annotations.filter( token =>
        token.annotatorType == AnnotatorType.TOKEN &&
          token.begin >= sentence.begin &&
            token.end <= sentence.end)

      val foundTokens = $$(searchTrie).search(tokens.map(_.result)).toList

      val finalTokens = if($(mergeOverlapping)) merge(foundTokens) else foundTokens

      for ((begin, end) <- finalTokens) {

        val firstTokenBegin = tokens(begin).begin
        val lastTokenEnd = tokens(end).end

        /** token indices are not relative to sentence but to document, adjust offset accordingly */
        val normalizedText = if(!$(buildFromTokens)) sentence.result.substring(firstTokenBegin  - sentence.begin, lastTokenEnd - sentence.begin + 1)
        else tokens.filter(t => t.begin >= firstTokenBegin && t.end <= lastTokenEnd).map(_.result).mkString(" ")


        val annotation = Annotation(
          outputAnnotatorType,
          firstTokenBegin,
          lastTokenEnd,
          normalizedText,
          Map("entity"->$(entityValue), "sentence" -> sentenceIndex.toString, "chunk" -> result.length.toString)
        )

        result.append(annotation)
      }
    }

    result
  }

}

trait ReadablePretrainedTextMatcher extends ParamsAndFeaturesReadable[TextMatcherModel] with HasPretrained[TextMatcherModel] {
  override val defaultModelName = None
  override def pretrained(): TextMatcherModel = super.pretrained()
  override def pretrained(name: String): TextMatcherModel = super.pretrained(name)
  override def pretrained(name: String, lang: String): TextMatcherModel = super.pretrained(name, lang)
  override def pretrained(name: String, lang: String, remoteLoc: String): TextMatcherModel = super.pretrained(name, lang, remoteLoc)
}

object TextMatcherModel extends ReadablePretrainedTextMatcher