package com.johnsnowlabs.nlp.annotators.tokenizer.wordpiece

import com.johnsnowlabs.nlp.annotators.common.{Sentence, SentenceSplit, WordpieceTokenized}
import org.scalatest.FlatSpec

class WordpieceTestSpec extends FlatSpec {
  val basicTokenizer = new BasicTokenizer()

  // Test vocabulary
  val pieces = Array("I", "un", "##am", "##bi", "##gouos", "##ly", "good", "!", "[UNK]", "[CLS]", "[SEP]")
  val vocabulary = pieces.zipWithIndex.toMap

  "isPunctuation" should "detect punctuation chars" in {
    val punctuations = ",:\";~`'-"
    val nonPunctuations = "aA12zZ \n\r\t"

    for (p <- punctuations) {
      assert(basicTokenizer.isPunctuation(p))
    }

    for (p <- nonPunctuations) {
      assert(!basicTokenizer.isPunctuation(p))
    }
  }

  "isWhitespace" should "detect whitespace chars" in {
    val whitespaces = " \n\r\t"
    val nonWhitespaces = "-=';\""

    for (p <- whitespaces) {
      assert(basicTokenizer.isWhitespace(p))
    }

    for (p <- nonWhitespaces) {
      assert(!basicTokenizer.isWhitespace(p))
    }
  }

  "stripAccents" should "work correct" in {
    val test =   "¡¢£¤¥¦§¨©ª«¬®¯°±²³´µ¶·¸¹º»¼½¾ ¿ÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓ ÔÕÖ×ØÙÚÛ ÜÝ Þßàáâãäåæçèéêëìíîïð ñòóôõö÷ø ùúûüýþÿ"
    val result = "¡¢£¤¥¦§¨©ª«¬®¯°±²³´µ¶·¸¹º»¼½¾ ¿AAAAAAÆCEEEEIIIIÐNOO OOO×ØUUU UY Þßaaaaaaæceeeeiiiið nooooo÷ø uuuuyþy"

    assert(basicTokenizer.stripAccents(test) == result)
    assert(basicTokenizer.normalize(test) == result.toLowerCase)
  }

  "isChinese" should "work correct" in {
    val chinese = "转注字轉注形声字形聲字"
    val nonChinese = "¹º»¼½¾¿ÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓ ÔÕAa123"

    for (p <- chinese) {
      assert(basicTokenizer.isChinese(p))
    }

    for (p <- nonChinese) {
      assert(!basicTokenizer.isChinese(p))
    }
  }

  "tokenization" should "tokenize sentence" in {
    val text = "Hello, I won't be from New York in the U.S.A. (and you know it héroe). " +
      "Give me my horse! or $100\" +\n    \" bucks 'He said', I'll defeat markus-crassus. You understand. Goodbye George E. Bush. www.google.com."

    val sentence = Sentence(text, 10, text.length + 9)
    val result = Array("Hello", ",", "I", "won", "'", "t", "be", "from", "New", "York", "in", "the", "U", ".", "S", ".",
      "A", ".", "(", "and", "you", "know","it","heroe", ")", ".", "Give", "me", "my", "horse", "!", "or",
      "$", "100", "\"", "+", "\"", "bucks", "'", "He", "said", "'", ",", "I", "'", "ll", "defeat", "markus",
      "-", "crassus", ".", "You", "understand", ".", "Goodbye", "George", "E", ".", "Bush", ".", "www", ".",
      "google", ".", "com", ".")

    val tokenizer = new BasicTokenizer(lowerCase = false)
    val tokens = tokenizer.tokenize(sentence)
    // 1. Check number of tokens
    assert(tokens.length == result.length)

    val lowercaseTokens = basicTokenizer.tokenize(sentence)
    // 2. Check that lowercased returns the same number of tokens
    assert(lowercaseTokens.length == result.length)

    var index = 0
    for (((correct, token), lowercase) <- result zip tokens zip lowercaseTokens) {
      // 3. Check token text
      assert(token.token == correct)
      assert(lowercase.token == correct.toLowerCase)

      val newIndex = text.indexOf(correct, index)
      index = if (newIndex >= 0) newIndex else index + 1

      // 4. Check token's begin and end
      assert(token.begin == index + 10)
      assert(token.end == index + correct.length - 1 + 10)

      // 5. Check lowercased's begin and end
      assert(lowercase.begin == token.begin)
      assert(lowercase.end == token.end)

      index = token.end + 1 - 10
    }
  }

  "tokenization" should "tokenize chinese text correct" in {
    val text = "Hello注形声sd,~ and bye!"
    val sentence = Sentence(text, 0, text.length - 1)
    val result = Array("Hello", "注", "形", "声", "sd", ",", "~", "and", "bye", "!")

    val tokenizer = new BasicTokenizer(lowerCase = false)
    val tokens = tokenizer.tokenize(sentence)

    assert(tokens.length == result.length)
    for ((token, expected) <- tokens zip result) {
      assert(token.token == expected)
    }
  }

  "wordpiece" should "encode words correct" in {
    val text = "I unambigouosly good 3Asd!"
    val sentence = Sentence(text, 0, text.length - 1)

    val expected = Array("I", "un", "##am", "##bi", "##gouos", "##ly", "good", "[UNK]", "!")

    val tokenizer = new WordpieceTokenizerModel()
        .setLowercase(false)
        .setVocabulary(vocabulary)

    val source = SentenceSplit.pack(Seq(sentence))
    val annotations = tokenizer.annotate(source)

    val tokens = WordpieceTokenized.unpack(annotations ++ source).head.tokens

    for ((token, correct) <- tokens zip expected) {
      // Check wordpiece
      assert(token.wordpiece == correct)
      assert(token.pieceId == vocabulary(token.wordpiece))

      // Check isWordStart
      val isWordStart = !correct.startsWith("##")
      assert(token.isWordStart == isWordStart)
    }
  }
}
