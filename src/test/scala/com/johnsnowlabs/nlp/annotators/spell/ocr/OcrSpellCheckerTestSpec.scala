package com.johnsnowlabs.nlp.annotators.spell.ocr
import com.github.liblevenshtein.transducer.{Candidate, ITransducer}
import com.johnsnowlabs.nlp.annotators.spell.ocr.parser._
import com.johnsnowlabs.nlp.{Annotation, SparkAccessor}
import org.scalatest._

//import scala.collection.mutable.Map


object MedicationClass extends VocabParser {

  override val vocab = loadCSV("meds_wcase.txt")

  override val transducer: ITransducer[Candidate] = generateTransducer

  override val label: String = "_MED_"

  override val maxDist: Int = 3

}

object AgeToken extends RegexParser {

  override val regex: String = "1?[0-9]{0,2}-(year|month|day)(s)?(-old)?"

  override val transducer: ITransducer[Candidate] = generateTransducer

  override val label: String = "_AGE_"

  override val maxDist: Int = 2

}


object UnitToken extends VocabParser {

  override val vocab: Set[String] = Set("MG=", "MEQ=", "TAB",
    "tablet", "mmHg", "TMIN", "TMAX", "mg/dL", "MMOL/L", "mmol/l", "mEq/L", "mmol/L",
    "mg", "ml", "mL", "mcg", "mcg/", "gram", "unit", "units", "DROP", "intl", "KG")

  override val transducer: ITransducer[Candidate] = generateTransducer

  override val label: String = "_UNIT_"

  override val maxDist: Int = 3

}


class OcrSpellCheckerTestSpec extends FlatSpec {

  /* TODO  Note some test cases should be moved to internal repo */

  trait Scope extends TokenClasses {
    weights += ('l' -> Map('1' -> 0.5f, '!' -> 0.2f), 'P' -> Map('F' -> 0.2f))
  }

  "weighted Levenshtein distance" should "produce weighted results" in new Scope {
    assert(wLevenshteinDist("c1ean", "clean") > wLevenshteinDist("c!ean", "clean"))
    assert(wLevenshteinDist("crean", "clean") > wLevenshteinDist("c!ean", "clean"))
    assert(wLevenshteinDist("Fatient", "Patient") < wLevenshteinDist("Aatient", "Patient"))
  }


  "a model" should "serialize properly" in {

    val trainCorpusPath = "../auxdata/spell_dataset/vocab/bigone.txt"
    val langModelPath = "../auxdata/"

    import SparkAccessor.spark.implicits._
    val ocrSpellModel = new OcrSpellCheckApproach().
      setInputCols("text").
      setTrainCorpusPath(trainCorpusPath).
      //setSpecialClasses(List(DateToken, NumberToken, AgeToken, UnitToken, MedicationClass)).
      setSpecialClasses(List.empty).
      train(Seq.empty[String].toDF("text"))

    ocrSpellModel.readModel(langModelPath, SparkAccessor.spark, "")
    ocrSpellModel.write.overwrite.save("./test_spell_checker")
    OcrSpellCheckModel.read.load("./test_spell_checker")
  }

  // TODO: move this logic to spark-nlp-models
  "a model" should "train and predict" in {

    val trainCorpusPath = "../auxdata/spell_dataset/vocab/bigone.txt"
    val langModelPath = "../auxdata/spell_dataset"

    import SparkAccessor.spark.implicits._
    val ocrspell = new OcrSpellCheckApproach().
                    setInputCols("text").
                    setTrainCorpusPath(trainCorpusPath).
                    setSpecialClasses(List(DateToken, NumberToken, AgeToken, UnitToken, MedicationClass)).
                    fit(Seq.empty[String].toDF("text"))

    ocrspell.readModel(langModelPath, SparkAccessor.spark, "")

    ocrspell.annotate(Seq(Annotation(" He also complains of \" bene pain ’ . He denies having any fevers or chills . He deries having" +
                 " any chest pain , palpitalicns , He denies any worse sxtramity"))).foreach(println)
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
    assert(vocab._1.size == 10)
  }


  "age classes" should "recognize different age patterns" in {
    import scala.collection.JavaConversions._
    val transducer = AgeToken.generateTransducer

    assert(transducer.transduce("102-year-old").toList.exists(_.distance == 0))
    assert(transducer.transduce("102-years-old").toList.exists(_.distance == 0))
    assert(transducer.transduce("24-year-old").toList.exists(_.distance == 0))
    assert(transducer.transduce("24-months-old").toList.exists(_.distance == 0))
    assert(transducer.transduce("24-month-old").toList.exists(_.distance == 0))
    assert(transducer.transduce("4-days-old").toList.exists(_.distance == 0))
    assert(transducer.transduce("14-year").toList.exists(_.distance == 0))
    assert(transducer.transduce("14-years").toList.exists(_.distance == 0))


  }

  "medication classes" should "recognize different medications patterns" in {
    import scala.collection.JavaConversions._
    val transducer = MedicationClass.generateTransducer

    assert(transducer.transduce("Eliquis").toList.exists(_.distance == 0))
    assert(transducer.transduce("Percocet").toList.exists(_.distance == 0))

  }

  "number classes" should "recognize different medications patterns" in {
    import scala.collection.JavaConversions._
    val transducer = NumberToken.generateTransducer

    assert(transducer.transduce("100.3").toList.exists(_.distance == 0))
    assert(NumberToken.separate("$40,000").equals(NumberToken.label))
  }


  "date classes" should "recognize different date and time formats" in {
    import scala.collection.JavaConversions._
    val transducer = DateToken.generateTransducer

    assert(transducer.transduce("10/25/1982").toList.exists(_.distance == 0))
    assert(DateToken.separate("10/25/1982").equals(DateToken.label))
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
