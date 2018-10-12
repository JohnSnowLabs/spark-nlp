package com.johnsnowlabs.nlp.annotators.spell.ocr
import com.github.liblevenshtein.transducer.{Algorithm, Candidate}
import org.json4s.jackson.JsonMethods
import com.github.liblevenshtein.transducer.factory.TransducerBuilder
import com.johnsnowlabs.nlp.{Annotation, SparkAccessor}
import org.scalatest._

import scala.collection.mutable

class OcrSpellCheckerTestSpec extends FlatSpec {

  trait Scope extends TokenClasses {
    weights += ('l' -> Map('1' -> 0.5f, '!' -> 0.2f), 'P' -> Map('F' -> 0.2f))
  }

  "weighted Levenshtein distance" should "produce weighted results" ignore new Scope {
    assert(wLevenshteinDist("c1ean", "clean") > wLevenshteinDist("c!ean", "clean"))
    assert(wLevenshteinDist("crean", "clean") > wLevenshteinDist("c!ean", "clean"))
    assert(wLevenshteinDist("Fatient", "Patient") < wLevenshteinDist("Aatient", "Patient"))
  }

  "weighted Levenshtein distance" should "properly compute distance to a regular language - dates" in new Scope {
    assert(wLevenshteinDateDist("10/0772018") == 1.0f)
  }

  "levenshtein automaton" should "build index and search terms" ignore {
    import scala.collection.JavaConversions._
    // TODO move to resources
    val path = "/home/jose/auxdata/ocr_spell/"
    val filenames = mutable.Seq("meds.json", "pubmed_vocab.json").map(path + _)
    val vocab =  filenames.flatMap { filename =>
      val source = scala.io.Source.fromFile(filename)
      val json = JsonMethods.parse(source.reader())
      json.children.map(_.values.toString)
    }.toSet

    val transducer = new TransducerBuilder().
      dictionary(vocab.toList.sorted, true).
      algorithm(Algorithm.TRANSPOSITION).
      defaultMaxDistance(2).
      includeDistance(true).
      build

    val time = System.nanoTime()

    // TODO remove this!
    val testData = s"""fibrillation on Eliquis, history of Gi bleed, CR-M3, and anemia who presents with ull wegks
      oi generalized fatigue and fecling umvell. He also notes some shortness of breath and
      worsening dyspnea wilh minimal éxerlion. His major complaints are shoulder and joint
    pains, diffusely. He also complains of "bene pain’. He denies having any fevers or chills.
    He deries having any chest pain, palpitalicns, He denies any worse sxtramity""".split(" ")

    System.out.println(s"Done, ${(System.nanoTime() - time)/1e9}\n")
  }


  "a model" should "train and predict" in {

    import SparkAccessor.spark.implicits._
    val ocrspell = new OcrSpellCheckApproach().
                    setInputCols("text").
                    setTrainCorpusPath("../auxdata/spell_dataset/vocab/spell_corpus.txt").
                    setVocabPath("../auxdata/spell_dataset/vocab/vocab").
                    train(Seq.empty[String].toDF("text"))

    val result = ocrspell.annotate(Seq(
      Annotation("swelling than his baseline. He states ha's teen compliant with his medications. Although he stales he ran out of his Eliquis & few wesks ago. He denies having any blood in his stools or meiena, although he does 1ake iron pills and states his stools arc frequently black His hemoglobin is a1 baseline."),
      Annotation("(01712/1982),"), Annotation("duodenojejunostom1,")))

    assert(result.exists(_.result == "duodenojejunostomy,"))
    assert(result.exists(_.result == "(01/12/1982)"))

  }

  "a spell checker" should "correclty handle dates" ignore {

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
    }*/

/*
  "jsl" should "handle commas at the end" in {
    //DictWord.setDict(null)
    val result = BaseParser.parse("thermalis,")
    result.foreach(println)
  }*/

}
