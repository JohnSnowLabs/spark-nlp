package com.johnsnowlabs.nlp.annotators.spell.context
import com.johnsnowlabs.nlp.annotators.common.{PrefixedToken, SuffixedToken}
import com.johnsnowlabs.nlp.annotators.{Normalizer, Tokenizer}
import com.johnsnowlabs.nlp.annotators.spell.context.parser._
import com.johnsnowlabs.nlp.{Annotation, DocumentAssembler, SparkAccessor}
import org.apache.spark.ml.Pipeline
import org.scalatest._


class ContextSpellCheckerTestSpec extends FlatSpec {


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
    val spellChecker = new ContextSpellCheckerApproach().
      setTrainCorpusPath("src/test/resources/test.txt").
      setSuffixes(Array(".", ":", "%", ",", ";", "?", "'", "!”", "”", "!”", ",”", ".”")).
      setPrefixes(Array("'", "“", "“‘")).
      setMinCount(1.0)

    val (map, _) = spellChecker.genVocab(path)
    assert(map.exists(_._1.equals("seed")))
    assert(map.exists(_._1.equals("”")))
    assert(map.exists(_._1.equals("“")))

  }


  "a Spell Checker" should "work in a pipeline with Tokenizer" in {
    import SparkAccessor.spark
    import spark.implicits._

    val data = Seq("It was a cold , dreary day and the country was white with smow .",
      "He wos re1uctant to clange .",
      "he is gane .").toDF("text")

    val documentAssembler =
      new DocumentAssembler().
        setInputCol("text").
        setOutputCol("doc")

    val tokenizer: Tokenizer = new Tokenizer()
      .setInputCols(Array("doc"))
      .setOutputCol("token")

    val normalizer: Normalizer = new Normalizer()
      .setInputCols(Array("token"))
      .setOutputCol("normalized")

    val spellChecker = ContextSpellCheckerModel
      .pretrained()
      .setTradeOff(12.0f)
      .setInputCols("token")
      .setOutputCol("checked")

    val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, spellChecker)).fit(data)
    pipeline.transform(data).select("checked").show(truncate=false)

  }

  "a Spell Checker" should "correctly handle paragraphs defined by newlines" in {
    import SparkAccessor.spark
    import spark.implicits._

    val data = Seq("Incruse Ellipta, 1 PUFF, Inhalation,\nQAM\n\nlevothyroxine 50 meg (0.05 mg) oral\ntablet, See Instructions\n\nlisinopril 20 mg oral tablet, See\nInstructions, 5 refills\n\nloratadine 10 mg oral tablet, 10 MG=\n1 TAB, PO, Dally\n\nPercocet 10/325 oral tablet, 2 TAB,\nPO, TID, PRN").toDF("text")

    val documentAssembler =
      new DocumentAssembler().
        setInputCol("text").
        setOutputCol("doc").
        setTrimAndClearNewLines(false)

    val tokenizer: Tokenizer = new Tokenizer()
      .setInputCols(Array("doc"))
      .setOutputCol("token")
      .setIncludeDefaults(false)
      .setTargetPattern("[a-zA-Z0-9]+|\n|\n\n|\\(|\\)|\\.|\\,")

    val normalizer: Normalizer = new Normalizer()
      .setInputCols(Array("token"))
      .setOutputCol("normalized")

    val spellChecker = ContextSpellCheckerModel
      .pretrained()
      .setTradeOff(12.0f)
      .setInputCols("token")
      .setOutputCol("checked")
      .setUseNewLines(true)

    val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, spellChecker)).fit(data)
    pipeline.transform(data).select("checked").show(truncate=false)

  }



  "a model" should "serialize properly" in {

    val trainCorpusPath = "../auxdata/spell_dataset/vocab/bigone.txt"
    val langModelPath = "../auxdata/"

    import SparkAccessor.spark.implicits._
    val ocrSpellModel = new ContextSpellCheckerApproach().
      setInputCols("text").
      setTrainCorpusPath(trainCorpusPath).
      setSpecialClasses(List(DateToken, NumberToken)).
      fit(Seq.empty[String].toDF("text"))

    ocrSpellModel.readModel(langModelPath, SparkAccessor.spark, "", useBundle=true)

    ocrSpellModel.write.overwrite.save("./test_spell_checker")
    val loadedModel = ContextSpellCheckerModel.read.load("./test_spell_checker")
    val testStr = "He deries having any chest pain , palpitalicns , He denies any worse sxtramity"
    val annotations = testStr.split(" ").map(Annotation(_)).toSeq
    loadedModel.annotate(annotations).foreach(println)
  }

  "a spell checker" should "correclty parse training data" in {
    val ocrspell = new ContextSpellCheckerApproach().
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
