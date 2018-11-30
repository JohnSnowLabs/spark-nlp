package com.johnsnowlabs.nlp.annotators.spell.ocr
import com.github.liblevenshtein.transducer.{Candidate, ITransducer}
import com.johnsnowlabs.nlp.annotators.{Normalizer, Tokenizer}
import com.johnsnowlabs.nlp.annotators.spell.ocr.parser._
import com.johnsnowlabs.nlp.util.io.OcrHelper
import com.johnsnowlabs.nlp.{Annotation, DocumentAssembler, LightPipeline, SparkAccessor}
import org.apache.spark.ml.Pipeline
import org.scalatest._


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

  trait Scope extends WeightedLevenshtein {
    val weights = Map('l' -> Map('1' -> 0.5f, '!' -> 0.2f), 'P' -> Map('F' -> 0.2f))
  }

  "weighted Levenshtein distance" should "produce weighted results" in new Scope {
    assert(wLevenshteinDist("c1ean", "clean", weights) > wLevenshteinDist("c!ean", "clean", weights))
    assert(wLevenshteinDist("crean", "clean", weights) > wLevenshteinDist("c!ean", "clean", weights))
    assert(wLevenshteinDist("Fatient", "Patient", weights) < wLevenshteinDist("Aatient", "Patient", weights))
  }


  "a Spell Checker" should "correctly preprocess training data" in {

    val path = "src/test/resources/test.txt"
    val spellChecker = new OcrSpellCheckApproach().
      setTrainCorpusPath("src/test/resources/test.txt").
      setSuffixes(Array(".", ":", "%", ",", ";", "?", "'", "!”", "”", "!”", ",”", ".”")).
      setPrefixes(Array("'", "“", "“‘")).
      setMinCount(1.0)

    val (map, _) = spellChecker.genVocab(path)
    assert(map.contains("seed"))
    assert(map.contains("”"))
    assert(map.contains("“"))

  }

  // TODO complete when we have a generic pre-trained model.
  "a Spell Checker" should "work in a pipeline with Tokenizer" in {
    import SparkAccessor.spark
    import spark.implicits._

    val data = OcrHelper.createDataset(spark,
      "ocr/src/test/resources/pdfs/h_and_p.pdf",
      "region", "metadata")

    val documentAssembler =
      new DocumentAssembler().
        setInputCol("region").
        setMetadataCol("metadata")

    val tokenizer: Tokenizer = new Tokenizer()
      .setInputCols(Array("sentence"))
      .setOutputCol("token")

    val normalizer: Normalizer = new Normalizer()
      .setInputCols(Array("token"))
      .setOutputCol("normalized")

    val pipeline = new Pipeline().setStages(Array(documentAssembler)).fit(Seq.empty[String].toDF("region"))
    pipeline.transform(data).show()

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
      fit(Seq.empty[String].toDF("text"))

    ocrSpellModel.readModel(langModelPath, SparkAccessor.spark, "", useBundle=true)

    ocrSpellModel.write.overwrite.save("./test_spell_checker")
    val loadedModel = OcrSpellCheckModel.read.load("./test_spell_checker")

    loadedModel.annotate(Seq(Annotation("He also complains of \" bene pain ’ . He denies having any fevers or chills . He deries having" +
      " any chest pain , palpitalicns , He denies any worse sxtramity"))).foreach(println)
  }

  // TODO: move this training logic to spark-nlp-models
  "a model" should "train and predict" in {

    val trainCorpusPath = "../auxdata/spell_dataset/vocab/bigone.txt"
    val langModelPath = "../auxdata/"

    import SparkAccessor.spark.implicits._
    val ocrspell = new OcrSpellCheckApproach().
                    setInputCols("text").
                    setTrainCorpusPath(trainCorpusPath).
                    setSpecialClasses(List(DateToken, NumberToken, AgeToken, UnitToken, MedicationClass)).
                    fit(Seq.empty[String].toDF("text"))

    ocrspell.readModel(langModelPath, SparkAccessor.spark, "", true)

    ocrspell.annotate(Seq(Annotation("He also complains of \" bene pain ’ . He denies having any fevers or chills . He deries having" +
                 " any chest pain , palpitalicns , He denies any worse sxtramity"))).foreach(println)
  }


  "a spell checker" should "correclty parse training data" in {
    val ocrspell = new OcrSpellCheckApproach().
      setMinCount(1.0)
    val trainCorpusPath = "src/main/resources/spell_corpus.txt"
    ocrspell.setTrainCorpusPath(trainCorpusPath)
    val (vocab, classes) = ocrspell.genVocab(trainCorpusPath)
    val vMap = vocab.toMap
    val totalTokenCount = 55.0
    assert(vocab.size == 35)
    assert(vMap.getOrElse("_EOS_", 0.0) == math.log(3.0) - math.log(totalTokenCount), "Three sentences should cause three _BOS_ markers")
    assert(vMap.getOrElse("_BOS_", 0.0) == math.log(3.0) - math.log(totalTokenCount), "Three sentences should cause three _EOS_ markers")

    assert(classes.size == 35, "")

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

  "number classes" should "recognize different number patterns" in {
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

  "suffixes and prefixes" should "recognized and handled properly" in {
    val suffixedToken = SuffixedToken(Array(")", ","))
    val prefixedToken = PrefixedToken(Array("("))

    var tmp = suffixedToken.separate("People,")
    assert(tmp.equals("People ,"))

    tmp = prefixedToken.separate(suffixedToken.separate("(08/10/1982)"))
    assert(tmp.equals("( 08/10/1982 )"))

  }


}
