package com.johnsnowlabs.nlp.annotators.tokenizer.bpe

import com.johnsnowlabs.nlp.annotators.common.{Sentence, TokenPiece}
import com.johnsnowlabs.tags.FastTest
import org.scalatest.FlatSpec


class XlmTokenizerTestSpec extends FlatSpec {
  val vocab: Map[String, Int] =
    Array(
      "<s>",
      "</s>",
      "<unk>",
      "<pad>",
      "<special0>",
      "<special1>",
      "<special2>",
      "<special3>",
      "<special4>",
      "<special5>",
      "<special6>",
      "<special7>",
      "<special8>",
      "<special9>",
      "i</w>",
      "un",
      "ambi",
      "gou",
      "os",
      "ly</w>",
      "good</w>",
      "3",
      "as",
      "d</w>",
      "!</w>"
    ).zipWithIndex.toMap

  val merges: Map[(String, String), Int] = Array(
    "u n",
    "a m",
    "a s",
    "o u",
    "b i",
    "o s",
    "i g",
    "n a",
    "am b",
    "g o",
    "s l",
    "o d</w>",
    "l y</w>",
    "am bi",
    "g ou",
    "m b",
    "go od</w>",
    "bi g",
    "s d</w>",
    "o o"
  ).map(_.split(" ")).map { case Array(c1, c2) => (c1, c2) }.zipWithIndex.toMap

  private def assertEncodedCorrectly(text: String,
                                     encoded: Array[TokenPiece],
                                     expected: Array[String],
                                     expectedIds: Array[Int]): Unit = {
    //    println(encoded.mkString("Array(\n  ", ",\n  ", "\n)"))
    for (i <- encoded.indices) {
      val piece = encoded(i)
      assert(piece.wordpiece == expected(i))
      assert(piece.pieceId == expectedIds(i))

      assert(text.slice(piece.begin, piece.end + 1).toLowerCase == piece.wordpiece)
    }
  }

  val xlmTokenizer: BpeTokenizer = BpeTokenizer.forModel("xlm", merges, vocab)

  "XlmTokenizer" should "encode words correctly" taggedAs FastTest in {
    val text = "I unambigouosly good 3Asd!"
    val sentence = Sentence(text, 0, text.length - 1, 0)

    val expected: Array[String] = Array(
      "i", "un", "ambi", "gou", "os", "ly", "good", "3", "as", "d", "!"
    )
    val expectedIds: Array[Int] = Array(14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24)

    val tokenized = xlmTokenizer.tokenize(sentence)

    val encoded = xlmTokenizer.encode(tokenized)

    assertEncodedCorrectly(text, encoded, expected, expectedIds)
  }

  "XlmTokenizer" should "encode sentences with special tokens correctly" taggedAs FastTest in {
    val text = "I unambigouosly <special1> 3Asd!"
    val sentence = Sentence(text, 0, text.length - 1, 0)

    val expected = Array("i", "un", "ambi", "gou", "os", "ly", "<special1>", "3", "as", "d", "!")
    val expectedIds = Array(14, 15, 16, 17, 18, 19, 5, 21, 22, 23, 24)

    val tokenizedWithMask = xlmTokenizer.tokenize(sentence)
    val encoded = xlmTokenizer.encode(tokenizedWithMask)

    assertEncodedCorrectly(text, encoded, expected, expectedIds)
  }

  "XlmTokenizer" should "handle empty sentences" taggedAs FastTest in {
    val text = " \n"
    val sentence = Sentence(text, 0, text.length - 1, 0)

    val tokenized = xlmTokenizer.tokenize(sentence)
    val encoded = xlmTokenizer.encode(tokenized)
    assert(tokenized.isEmpty)
    assert(encoded.isEmpty)
  }

  "XlmTokenizer" should "add sentence padding correctly if requested" taggedAs FastTest in {
    val tokenizer = BpeTokenizer.forModel("xlm", merges, vocab, padWithSentenceTokens = true)

    val text = "I unambigouosly <special1> 3Asd!"
    val sentence = Sentence(text, 0, text.length - 1, 0)

    val expected = Array("<s>", "i", "un", "ambi", "gou", "os", "ly", "<special1>", "3", "as", "d", "!", "</s>")
    val expectedIds = Array(0, 14, 15, 16, 17, 18, 19, 5, 21, 22, 23, 24, 1)

    val tokenized = tokenizer.tokenize(sentence)
    val encoded = tokenizer.encode(tokenized)

    val textPadded = "<s>" + text + "</s>"
    assertEncodedCorrectly(textPadded, encoded, expected, expectedIds)

    assert(tokenized.head.token == "<s>")
    assert(tokenized.last.token == "</s>")

  }

  "XlmTokenizer" should "handle unknown words" taggedAs FastTest in {
    val text = "???"
    val sentence = Sentence(text, 0, text.length - 1, 0)

    val tokenized = xlmTokenizer.tokenize(sentence)
    val encoded = xlmTokenizer.encode(tokenized)
    assert(encoded.forall(_.pieceId == vocab("<unk>")))
  }
}
