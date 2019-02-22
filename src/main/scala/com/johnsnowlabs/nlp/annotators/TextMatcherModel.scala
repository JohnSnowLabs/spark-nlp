package com.johnsnowlabs.nlp.annotators

import com.johnsnowlabs.collections.SearchTrie
import com.johnsnowlabs.nlp._
import org.apache.spark.ml.util.Identifiable
import com.johnsnowlabs.nlp.AnnotatorType._
import com.johnsnowlabs.nlp.serialization.ArrayFeature
import org.apache.spark.ml.param.BooleanParam

import scala.collection.mutable.ArrayBuffer

/**
  * Extracts entities out of provided phrases
  * @param uid internally renquired UID to make it writable
  * @@ entitiesPath: Path to file with phrases to search
  * @@ insideSentences: Should Extractor search only within sentence borders?
  */
class TextMatcherModel(override val uid: String) extends AnnotatorModel[TextMatcherModel] {

  override val annotatorType: AnnotatorType = CHUNK

  override val requiredAnnotatorTypes: Array[AnnotatorType] = Array(DOCUMENT, TOKEN)

  val parsedEntities = new ArrayFeature[Array[String]](this, "parsedEntities")

  val caseSensitive = new BooleanParam(this, "caseSensitive", "whether to match regardless of case. Defaults true")

  def setEntities(value: Array[Array[String]]): this.type = set(parsedEntities, value)

  def setCaseSensitive(v: Boolean): this.type =
    set(caseSensitive, v)

  def getCaseSensitive: Boolean =
    $(caseSensitive)

  lazy val searchTrie = SearchTrie.apply($$(parsedEntities), $(caseSensitive))

  /** internal constructor for writabale annotator */
  def this() = this(Identifiable.randomUID("ENTITY_EXTRACTOR"))

  /**
    * Searches entities and stores them in the annotation
    * @return Extracted Entities
    */
  /** Defines annotator phrase matching depending on whether we are using SBD or not */
  override def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {

    val result = ArrayBuffer[Annotation]()

    val sentences = annotations.filter(_.annotatorType == AnnotatorType.DOCUMENT)

    sentences.zipWithIndex.foreach{case (sentence, sentenceIndex) => {

      val tokens = annotations.filter( token =>
        token.annotatorType == AnnotatorType.TOKEN &&
          token.begin >= sentence.begin &&
            token.end <= sentence.end)

      for ((begin, end) <- searchTrie.search(tokens.map(_.result))) {

        val firstTokenBegin = tokens(begin).begin
        val lastTokenEnd = tokens(end).end

        /** token indices are not relative to sentence but to document, adjust offset accordingly */
        val normalizedText = sentence.result.substring(firstTokenBegin  - sentence.begin, lastTokenEnd - sentence.begin + 1)

        val annotation = Annotation(
          annotatorType,
          firstTokenBegin,
          lastTokenEnd,
          normalizedText,
          Map("sentence" -> sentenceIndex.toString)
        )

        result.append(annotation)
      }
    }}

    result
  }

}

object TextMatcherModel extends ParamsAndFeaturesReadable[TextMatcherModel]