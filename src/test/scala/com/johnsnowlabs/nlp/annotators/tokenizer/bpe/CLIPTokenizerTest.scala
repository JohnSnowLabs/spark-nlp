package com.johnsnowlabs.nlp.annotators.tokenizer.bpe

import com.johnsnowlabs.nlp.annotators.common.Sentence
import com.johnsnowlabs.tags.FastTest
import org.scalatest.flatspec.AnyFlatSpec

class CLIPTokenizerTest extends AnyFlatSpec {

  it should "tokenize correctly" taggedAs FastTest in {

    val vocab: Map[String, Int] = Map(
      "a</w>" -> 320,
      "photo</w>" -> 1125,
      "of</w>" -> 539,
      "cat</w>" -> 2368,
      "<|startoftext|>" -> 49406,
      "<|endoftext|>" -> 49407)

    val merges: Map[(String, String), Int] = {
      Map(
        (("t", "o</w>"), 19),
        (("h", "o"), 94),
        (("p", "ho"), 304),
        (("pho", "to</w>"), 613),
        (("o", "f</w>"), 27),
        (("a", "t</w>"), 24),
        (("c", "at</w>"), 1856),
        (("d", "o"), 127),
        (("do", "g</w>"), 1417))
    }

    val bpeTokenizer =
      BpeTokenizer.forModel("clip", merges, vocab, alwaysAddPrefix = false)

    val text = "a photo of a cat"
    val indexedTokens =
      bpeTokenizer.tokenize(Sentence(text, 0, text.length, 0))

    val encodedTokens = bpeTokenizer.encode(indexedTokens)

    val expected = Seq(49406, 320, 1125, 539, 320, 2368, 49407)
    encodedTokens.map(_.pieceId).zip(expected).map { case (id, expectedId) =>
      assert(id == expectedId)
    }
  }
}
