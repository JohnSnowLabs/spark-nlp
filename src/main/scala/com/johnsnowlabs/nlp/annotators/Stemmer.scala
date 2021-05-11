package com.johnsnowlabs.nlp.annotators

import com.johnsnowlabs.nlp.{Annotation, AnnotatorModel, HasSimpleAnnotate}
import org.apache.spark.ml.param.Param
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}

import scala.language.postfixOps

/**
  * Created by alext on 10/23/16.
  */

/**
  * Hard stemming of words for cut-of into standard word references.
  * See [[ https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/annotators/StemmerTestSpec.scala ]] for examples of how to use this API
  * @param uid internal uid element for storing annotator into disk
  * @groupname anno Annotator types
  * @groupdesc anno Required input and expected output annotator types
  * @groupname Ungrouped Members
  * @groupname param Parameters
  * @groupname setParam Parameter setters
  * @groupname getParam Parameter getters
  * @groupname Ungrouped Members
  * @groupprio anno  1
  * @groupprio param  2
  * @groupprio Ungrouped 3
  * @groupprio setParam  4
  * @groupprio getParam  5
  * @groupdesc Parameters A list of (hyper-)parameter keys this annotator can take. Users can set and get the parameter values through setters and getters, respectively.
  */
class Stemmer(override val uid: String) extends AnnotatorModel[Stemmer] with HasSimpleAnnotate[Stemmer] {

  import com.johnsnowlabs.nlp.AnnotatorType._

  /** this is the language of the text. default is English
    *
    * @group param
    **/
  val language: Param[String] = new Param(this, "language", "this is the language of the text")
  setDefault(language, "english")

  /** Output annotator type : TOKEN
    *
    * @group anno
    **/
  override val outputAnnotatorType: AnnotatorType = TOKEN
  /** Input annotator type : TOKEN
    *
    * @group anno
    **/
  override val inputAnnotatorTypes: Array[AnnotatorType] = Array(TOKEN)

  /** this is the language of the text. default is English
    *
    * @group param
    **/
  def setLanguage(value: String): Stemmer = set(language, value)

  /** Language for text   */
  def getLanguage: String = $(language)

  def this() = this(Identifiable.randomUID("STEMMER"))

  /** one-to-one stem annotation that returns single hard-stem per token */
  override def annotate(annotations: Seq[Annotation]): Seq[Annotation] =
    annotations.map { tokenAnnotation =>
      val stem = EnglishStemmer.stem(tokenAnnotation.result)
      Annotation(
        outputAnnotatorType,
        tokenAnnotation.begin,
        tokenAnnotation.end,
        stem,
        tokenAnnotation.metadata
      )
    }

}

object Stemmer extends DefaultParamsReadable[Stemmer]

object EnglishStemmer {

  def stem(word: String): String = {
    // Deal with plurals and past participles
    var stem = new Word(word).applyReplaces(
      "sses" → "ss",
      "ies" → "i",
      "ss" → "ss",
      "s" → "")

    if ((stem matchedBy ((~v~) + "ed")) ||
      (stem matchedBy ((~v~) + "ing"))) {

      stem = stem.applyReplaces(~v~)("ed" → "", "ing" → "")

      stem = stem.applyReplaces(
        "at" → "ate",
        "bl" → "ble",
        "iz" → "ize",
        (~d and not(~L or ~S or ~Z)) → singleLetter,
        (m == 1 and ~o) → "e")
    } else {
      stem = stem.applyReplaces(((m > 0) + "eed") → "ee")
    }

    stem = stem.applyReplaces(((~v~) + "y") → "i")

    // Remove suffixes
    stem = stem.applyReplaces(m > 0)(
      "ational" → "ate",
      "tional" → "tion",
      "enci" → "ence",
      "anci" → "ance",
      "izer" → "ize",
      "abli" → "able",
      "alli" → "al",
      "entli" → "ent",
      "eli" → "e",
      "ousli" → "ous",
      "ization" → "ize",
      "ation" → "ate",
      "ator" → "ate",
      "alism" → "al",
      "iveness" → "ive",
      "fulness" → "ful",
      "ousness" → "ous",
      "aliti" → "al",
      "iviti" → "ive",
      "biliti" → "ble")

    stem = stem.applyReplaces(m > 0)(
      "icate" → "ic",
      "ative" → "",
      "alize" → "al",
      "iciti" → "ic",
      "ical" → "ic",
      "ful" → "",
      "ness" → "")

    stem = stem.applyReplaces(m > 1)(
      "al" → "",
      "ance" → "",
      "ence" → "",
      "er" → "",
      "ic" → "",
      "able" → "",
      "ible" → "",
      "ant" → "",
      "ement" → "",
      "ment" → "",
      "ent" → "",
      ((~S or ~T) + "ion") → "",
      "ou" → "",
      "ism" → "",
      "ate" → "",
      "iti" → "",
      "ous" → "",
      "ive" → "",
      "ize" → "")

    // Tide up a little bit
    stem = stem applyReplaces(((m > 1) + "e") → "",
      (((m == 1) and not(~o)) + "e") → "")

    stem = stem applyReplaces ((m > 1 and ~d and ~L) → singleLetter)

    stem.toString
  }

  /**
    * Pattern that is matched against the word.
    * Usually, the end of the word is compared to suffix,
    * and the beginning is checked to satisfy a condition.

    */
  private case class Pattern(condition: Condition, suffix: String)

  /**
    * Condition, that is checked against the beginning of the word
    * Predicate to be applied to the word
    */
  private case class Condition(predicate: Word ⇒ Boolean) {
    def + = new Pattern(this, _: String)

    def unary_~ = this // just syntactic sugar

    def ~ = this

    def and(condition: Condition) = Condition((word) ⇒ predicate(word) && condition.predicate(word))

    def or(condition: Condition) = Condition((word) ⇒ predicate(word) || condition.predicate(word))
  }

  private def not: Condition ⇒ Condition = {
    case Condition(predicate) ⇒ Condition(!predicate(_))
  }

  private val emptyCondition = Condition(_ ⇒ true)

  private object m {
    def >(measure: Int) = Condition(_.measure > measure)

    def ==(measure: Int) = Condition(_.measure == measure)
  }

  private val S = Condition(_ endsWith "s")
  private val Z = Condition(_ endsWith "z")
  private val L = Condition(_ endsWith "l")
  private val T = Condition(_ endsWith "t")

  private val d = Condition(_.endsWithCC)

  private val o = Condition(_.endsWithCVC)

  private val v = Condition(_.containsVowels)

  /**
    * Builder of the stem
    * @param build Function to be called to build a stem
    *////////////////////////CLASS BEGINS HERE////////////////////////////////
  private case class StemBuilder(build: Word ⇒ Word)

  private def suffixStemBuilder(suffix: String) = StemBuilder(_ + suffix)

  private val singleLetter = StemBuilder(_ trimSuffix 1)

  private class Word(string: String) {
    val word = string.toLowerCase

    def trimSuffix(suffixLength: Int) = new Word(word substring (0, word.length - suffixLength))

    def endsWith = word endsWith _

    def +(suffix: String) = new Word(word + suffix)

    def satisfies = (_: Condition).predicate(this)

    def hasConsonantAt(position: Int): Boolean =
      (word.indices contains position) && (word(position) match {
        case 'a' | 'e' | 'i' | 'o' | 'u' ⇒ false
        case 'y' if hasConsonantAt(position - 1) ⇒ false
        case _ ⇒ true
      })

    def hasVowelAt = !hasConsonantAt(_: Int)

    def containsVowels = word.indices exists hasVowelAt

    def endsWithCC =
      (word.length > 1) &&
        (word(word.length - 1) == word(word.length - 2)) &&
        hasConsonantAt(word.length - 1)

    def endsWithCVC =
      (word.length > 2) &&
        hasConsonantAt(word.length - 1) &&
        hasVowelAt(word.length - 2) &&
        hasConsonantAt(word.length - 3) &&
        !(Set('w', 'x', 'y') contains word(word.length - 2))

    /**
      * Measure of the word -- the number of VCs
      * @return integer
      */
    def measure = word.indices.filter(pos ⇒ hasVowelAt(pos) && hasConsonantAt(pos + 1)).length

    def matchedBy: Pattern ⇒ Boolean = {
      case Pattern(condition, suffix) ⇒
        endsWith(suffix) && (trimSuffix(suffix.length) satisfies condition)
    }

    def applyReplaces(replaces: (Pattern, StemBuilder)*): Word = {
      for ((pattern, stemBuilder) ← replaces if matchedBy(pattern))
        return stemBuilder build trimSuffix(pattern.suffix.length)
      this
    }

    def applyReplaces(commonCondition: Condition)(replaces: (Pattern, StemBuilder)*): Word =
      applyReplaces(replaces map {
        case (Pattern(condition, suffix), stemBuilder) ⇒
          (Pattern(commonCondition and condition, suffix), stemBuilder)
      }: _*)

    override def toString = word
  }
  ////////////////////CLASS ENDS/////////////////////////////////
  private implicit def pimpMyRule[P <% Pattern, SB <% StemBuilder]
  (rule: (P, SB)): (Pattern, StemBuilder) = (rule._1, rule._2)
  private implicit def emptyConditionPattern: String ⇒ Pattern = Pattern(emptyCondition, _)
  private implicit def emptySuffixPattern: Condition ⇒ Pattern = Pattern(_, "")
  private implicit def suffixedStemBuilder: String ⇒ StemBuilder = suffixStemBuilder
}
