package com.johnsnowlabs.nlp.annotators.spell.ocr
import com.johnsnowlabs.nlp.{Annotation, SparkAccessor}
import org.scalatest._


class OcrSpellCheckerTestSpec extends FlatSpec {

  trait Scope extends TokenClasses {
    weights += ('l' -> Map('1' -> 0.5f, '!' -> 0.2f), 'P' -> Map('F' -> 0.2f))
  }

  // TODO: come back to this when scoring the final list with weighted distance
  "weighted Levenshtein distance" should "produce weighted results" ignore new Scope {
    assert(wLevenshteinDist("c1ean", "clean") > wLevenshteinDist("c!ean", "clean"))
    assert(wLevenshteinDist("crean", "clean") > wLevenshteinDist("c!ean", "clean"))
    assert(wLevenshteinDist("Fatient", "Patient") < wLevenshteinDist("Aatient", "Patient"))
  }

  "weighted Levenshtein distance" should "properly compute distance to a regular language - dates" in new Scope {
    assert(wLevenshteinDateDist("10/0772018") == 1.0f)
  }

  "a model" should "train and predict" in {

    val trainCorpusPath = "../auxdata/spell_dataset/vocab/spell_corpus.txt"
    val vocabPath =  "../auxdata/spell_dataset/vocab/vocab"

    import SparkAccessor.spark.implicits._
    val ocrspell = new OcrSpellCheckApproach().
                    setInputCols("text").
                    setTrainCorpusPath(trainCorpusPath).


      train(Seq.empty[String].toDF("text"))

    val result = ocrspell.annotate(Seq(
      Annotation("swelling than his baseline . he states ha's teen compliant with his medications ." +
                " although he stales he ran out of his Eliquis & few wesks ago . he denies having any blood" +
                " in his stools or meiena , although he does 1ake iron pills and states his stools arc" +
                " frequently black his hemoglobin is a1 baseline .")))

    //model.predict(Array(Array("frequently", "black" ,"his", "hemoglobin", "is", "a1",  "baseline", ".").map(vocabIds.get).map(_.get)))

    result.map(_.result).foreach(println)

  }


  "a spell checker" should "handle commas and dates" ignore {
    //Annotation("(01712/1982),"), Annotation("duodenojejunostom1,")
    //assert(result.exists(_.result == "duodenojejunostomy,"))
    //assert(result.exists(_.result == "(01/12/1982)"))
  }

  "a spell checker" should "correclty parse training data" ignore {

    val ocrspell = new OcrSpellCheckApproach().
      setMinCount(1.0)
    val vocab = ocrspell.genVocab("src/main/resources/spell_corpus.txt")
    assert(vocab.size == 10)

  }

  /*
    "double quotes" should "be parsed correctly" in {
      val result = DoubleQuotes.splits("\"aspirin\"")
      result.foreach(println)

    }

    "a parser" should "recognize special tokens" in {
      DictWord.setDict(Seq())
      val result = BaseParser.parse("(08/10/1982),")
      result.foreach(println)
    }

    "a parser" should "handle commas at the end" in {
      //DictWord.setDict(null)
      val result = BaseParser.parse("thermalis,")
      result.foreach(println)
  }*/

}
