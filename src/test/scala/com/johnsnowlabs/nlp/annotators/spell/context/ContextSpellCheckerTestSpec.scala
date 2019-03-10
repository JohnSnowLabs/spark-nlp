package com.johnsnowlabs.nlp.annotators.spell.context
import com.johnsnowlabs.nlp.annotators.common.{PrefixedToken, SuffixedToken}
import com.johnsnowlabs.nlp.annotators.{Normalizer, Tokenizer}
import com.johnsnowlabs.nlp.annotators.spell.context.parser._
import com.johnsnowlabs.nlp.{DocumentAssembler, SparkAccessor}
import org.apache.spark.ml.Pipeline
import org.scalatest._


class ContextSpellCheckerTestSpec extends FlatSpec {

  trait Scope extends WeightedLevenshtein {
    val weights = Map("l" -> Map("1" -> 0.5f, "!" -> 0.2f), "P" -> Map("F" -> 0.2f))
  }

  trait distFile extends WeightedLevenshtein {
    val weights = loadWeights("src/test/resources/dist.psv")
  }

  "weighted Levenshtein distance" should "work from file" in new distFile {
    assert(wLevenshteinDist("water", "Water", weights) < 1.0f)
    assert(wLevenshteinDist("50,000", "50,C00", weights) < 1.0f)
  }


  "weighted Levenshtein distance" should "produce weighted results" in new Scope {
    assert(wLevenshteinDist("clean", "c1ean", weights) > wLevenshteinDist("clean", "c!ean", weights))
    assert(wLevenshteinDist("clean", "crean", weights) > wLevenshteinDist("clean", "c!ean", weights))
    assert(wLevenshteinDist("Patient", "Fatient", weights) < wLevenshteinDist("Patient", "Aatient", weights))
  }

  "weighted Levenshtein distance" should "handle insertions and deletions" in new Scope {
    // inserting an 'h' should be cheaper than inserting a 'b'
    assert(wLevenshteinDist("cleanh", "clean", weights) < wLevenshteinDist("cleanb", "clean", weights))
    // deleting an 'm' should be cheaper than deleting an 'n'
    assert(wLevenshteinDist("albu", "album", weights) < wLevenshteinDist("clea", "clean", weights))

  }

  "weighted Levenshtein distance" should "handle insertions and deletions on procedures" in new Scope {
    override val weights = loadWeights("src/test/resources/distance.psv")

    val cost1 = weights("F")("P") + weights("a")("e")
    assert(wLevenshteinDist("Procedure", "Frocedura", weights) == cost1)

    val cost2 = weights("v")("y") + weights("iƐ")("if")
    assert(wLevenshteinDist("qualifying", "qualiving", weights) == cost2)

    val cost3 = weights("a")("o") + weights("^Ɛ")("^t")
    assert(wLevenshteinDist("to", "a", weights) == cost3)
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

    import SparkAccessor.spark.implicits._
    import scala.collection.JavaConversions._

    val ocrSpellModel = ContextSpellCheckerModel
      .pretrained()

    ocrSpellModel.write.overwrite.save("./test_spell_checker")
    val loadedModel = ContextSpellCheckerModel.read.load("./test_spell_checker")

    assert(loadedModel.specialTransducers.getOrDefault.size == 2, "default pretrained should come with 2 classes")

    // cope with potential change in element order in list
    val sortedTransducers = loadedModel.specialTransducers.getOrDefault.sortBy(_.label)

    assert(sortedTransducers(0).label == "_DATE_")
    assert(sortedTransducers(0).generateTransducer.transduce("10710/2018", 1).map(_.term()).contains("10/10/2018"))

    assert(sortedTransducers(1).label == "_NUM_")
    assert(sortedTransducers(1).generateTransducer.transduce("50,C00", 1).map(_.term()).contains("50,000"))

    val trellis = Array(Array.fill(6)(("the", 0.8, "the")),
      Array.fill(6)(("end", 1.2, "end")), Array.fill(6)((".", 1.2, ".")))
    val (decoded, cost) = loadedModel.decodeViterbi(trellis)
    assert(decoded.deep.equals(Array("the", "end", ".").deep))

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
