package com.johnsnowlabs.nlp.annotators.tokenizer.bpe

import com.johnsnowlabs.nlp.annotators.common.{IndexedToken, Sentence, TokenPiece}
import com.johnsnowlabs.tags.FastTest
import org.scalatest.flatspec.AnyFlatSpec

trait BpeTokenizerBehaviours {
  this: AnyFlatSpec =>

  val replaceCharBeforeAssertion : Option[String] = None

  protected def assertEncodedCorrectly(text: String,
                                     encoded: Array[TokenPiece],
                                     expected: Array[String],
                                     expectedIds: Array[Int]): Unit = {

    assert(encoded.length == expected.length)
    for (i <- encoded.indices) {
      val piece = encoded(i)
      assert(piece.wordpiece == expected(i))
      assert(piece.pieceId == expectedIds(i))

      replaceCharBeforeAssertion match {
        case Some(decodeChar) =>
          assert(text.slice(piece.begin, piece.end + 1) == piece.wordpiece.replace(decodeChar, " "))
        case _ => assert(text.slice(piece.begin, piece.end + 1) == piece.wordpiece)
      }
    }
  }

  protected def tokenizeAndEncode(tokenizer: BpeTokenizer, text: String): (Array[IndexedToken], Array[TokenPiece]) = {
    val sentence = Sentence(text, 0, text.length - 1, 0)

    val tokenized = tokenizer.tokenize(sentence)
    val encoded = tokenizer.encode(tokenized)
    (tokenized, encoded)
  }

  def correctBpeTokenizer(tokenizer: BpeTokenizer,
                          text: String,
                          expected: Array[String],
                          expectedIds: Array[Int]
                         ): Unit = {

    it should "encode words correctly" taggedAs FastTest in {
      val (_, encoded: Array[TokenPiece]) = tokenizeAndEncode(tokenizer, text)
      assertEncodedCorrectly(text, encoded, expected, expectedIds)
    }

    it should "add sentence padding correctly if requested" taggedAs FastTest in {
      tokenizer.padWithSentenceTokens = true

      val (tokenized: Array[IndexedToken], encoded: Array[TokenPiece]) = tokenizeAndEncode(tokenizer, text)

      //      val textPadded = "<|endoftext|>I unambigouosly<|endoftext|>"
      //      assertEncodedCorrectly(textPadded, encoded, expected, expectedIds)

      assert(tokenized.head.token == tokenizer.sentencePadding._1)
      assert(tokenized.last.token == tokenizer.sentencePadding._2)

      assert(encoded.head.pieceId == tokenizer.specialTokens.sentenceStart.id)
      assert(encoded.last.pieceId == tokenizer.specialTokens.sentenceEnd.id)

      tokenizer.padWithSentenceTokens = false
    }

  }

  def correctBpeTokenizerInFringeSituations(tokenizer: BpeTokenizer, unknownTokenText: String = "???") {
    it should "handle empty sentences" taggedAs FastTest in {
      val text = " \n"
      val (tokenized: Array[IndexedToken], encoded: Array[TokenPiece]) = tokenizeAndEncode(tokenizer, text)
      assert(tokenized.isEmpty)
      assert(encoded.isEmpty)
    }

    it should "handle unknown words" taggedAs FastTest in {
      val text = unknownTokenText
      val (_, encoded: Array[TokenPiece]) = tokenizeAndEncode(tokenizer, text)

      assert(encoded.forall(_.pieceId == tokenizer.specialTokens.unk.id))
    }

    it should "handle unknown byte encodings" taggedAs FastTest in {
      val text = "I unambigouosly \u8216"
      val (_, encoded: Array[TokenPiece]) = tokenizeAndEncode(tokenizer, text)

      assert(encoded.last.pieceId == tokenizer.specialTokens.unk.id)
    }
  }

  def correctBpeTokenizerSpecialTokens(tokenizer: BpeTokenizer,
                                       text: String,
                                       expected: Array[String],
                                       expectedIds: Array[Int]
                                      ): Unit = {

    it should "encode special tokens correctly" taggedAs FastTest in {
      val (_, encoded: Array[TokenPiece]) = tokenizeAndEncode(tokenizer, text)
      assertEncodedCorrectly(text, encoded, expected, expectedIds)
    }
  }
}
