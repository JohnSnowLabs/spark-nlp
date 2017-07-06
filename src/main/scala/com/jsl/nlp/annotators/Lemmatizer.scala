package com.jsl.nlp.annotators

import com.jsl.nlp.annotators.param.{AnnotatorParam, SerializedAnnotatorComponent, WritableAnnotatorComponent}
import com.jsl.nlp.{Annotation, Annotator, Document}
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}

/**
  * Created by saif on 28/04/17.
  */

/**
  * Class to find standarized lemmas from words. Uses a user-provided or default dictionary.
  * @param uid required internal uid provided by constructor
  * @@ lemmaDict: A dictionary of predefined lemmas must be provided
  */
class Lemmatizer(override val uid: String) extends Annotator {

  /** Internal serialized type of a dictionary so the lemmatizer can be loaded from disk */
  protected case class SerializedDictionary(dict: Map[String, String]) extends SerializedAnnotatorComponent[LemmatizerDictionary] {
    override def deserialize: LemmatizerDictionary = {
      LemmatizerDictionary(dict)
    }
  }

  /** Internal representation of the dictionary to allow serialization of the dictionary to be saved on disk */
  protected case class LemmatizerDictionary(dict: Map[String, String]) extends WritableAnnotatorComponent {
    override def serialize: SerializedAnnotatorComponent[LemmatizerDictionary] =
      SerializedDictionary(dict)
  }

  val lemmaDict: AnnotatorParam[LemmatizerDictionary, SerializedDictionary] =
    new AnnotatorParam[LemmatizerDictionary, SerializedDictionary](this, "lemma dictionary", "provide a lemma dictionary")

  override val annotatorType: String = Lemmatizer.annotatorType

  /** Requires a tokenizer since words need to be split up in tokens */
  override var requiredAnnotatorTypes: Array[String] = Array(RegexTokenizer.annotatorType)

  def this() = this(Identifiable.randomUID(Lemmatizer.annotatorType))

  def getLemmaDict: Map[String, String] = $(lemmaDict).dict

  def setLemmaDict(dictionary: Map[String, String]): this.type = set(lemmaDict, LemmatizerDictionary(dictionary))

  /**
    * @return one to one annotation from token to a lemmatized word, if found on dictionary or leave the word as is
    */
  override def annotate(document: Document, annotations: Seq[Annotation]): Seq[Annotation] = {
    annotations.collect {
      case tokenAnnotation: Annotation if tokenAnnotation.annotatorType == RegexTokenizer.annotatorType =>
        val token = document.text.substring(tokenAnnotation.begin, tokenAnnotation.end)
        Annotation(
          annotatorType,
          tokenAnnotation.begin,
          tokenAnnotation.end,
          Map(token -> getLemmaDict.getOrElse(token, token))
        )
    }
  }

}

object Lemmatizer extends DefaultParamsReadable[Lemmatizer] {
  val annotatorType = "lemma"
}
