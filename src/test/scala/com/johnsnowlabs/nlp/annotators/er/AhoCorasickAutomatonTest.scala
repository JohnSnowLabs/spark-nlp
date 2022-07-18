package com.johnsnowlabs.nlp.annotators.er

import com.johnsnowlabs.nlp.Annotation
import com.johnsnowlabs.nlp.AnnotatorType.{CHUNK, TOKEN}
import com.johnsnowlabs.nlp.annotators.common.Sentence
import org.scalatest.flatspec.AnyFlatSpec

class AhoCorasickAutomatonTest extends AnyFlatSpec {

  "AhoCorasickAutomaton" should "build a matching machine for english" in {

    val englishAlphabet = "abcdefghijklmnopqrstuvwxyz,"
    val entityPatterns =
      Array(EntityPattern("Noun", Seq("he", "she")), EntityPattern("Pronoun", Seq("his", "hers")))
    val text = "she is over there, look his eyes met hers"
    val sentence = Sentence(text, 0, text.length, 0)
    val tokens = Map(
      2 -> Annotation(TOKEN, 0, 2, "she", Map()),
      5 -> Annotation(TOKEN, 4, 5, "is", Map()),
      10 -> Annotation(TOKEN, 7, 10, "over", Map()),
      16 -> Annotation(TOKEN, 12, 16, "there", Map()),
      17 -> Annotation(TOKEN, 17, 17, ",", Map()),
      22 -> Annotation(TOKEN, 19, 22, "look", Map()),
      26 -> Annotation(TOKEN, 24, 26, "his", Map()),
      31 -> Annotation(TOKEN, 28, 31, "eyes", Map()),
      35 -> Annotation(TOKEN, 33, 35, "met", Map()),
      40 -> Annotation(TOKEN, 37, 40, "hers", Map()))

    val automaton = new AhoCorasickAutomaton(englishAlphabet, entityPatterns)
    automaton.buildMatchingMachine()
    val actualOutput = automaton.searchWords(sentence, tokens)

    val expectedOutput = List(
      Annotation(CHUNK, 0, 2, "she", Map("entity" -> "Noun", "sentence" -> "0")),
      Annotation(CHUNK, 24, 26, "his", Map("entity" -> "Pronoun", "sentence" -> "0")),
      Annotation(CHUNK, 37, 40, "hers", Map("entity" -> "Pronoun", "sentence" -> "0")))
    assert(actualOutput == expectedOutput)
  }

  it should "build a matching machine for Japanese" in {
    val japaneseSampleAlphabet = "ざ音楽数ポ学生理学ぎ"
    val entityPatterns = Array(EntityPattern("Test-Japanese", Seq("音楽数", "生")))
    val text = "音楽数 学 生 理学"
    val sentence = Sentence(text, 0, text.length, 0)
    val tokens = Map(
      2 -> Annotation(TOKEN, 0, 2, "音楽数", Map()),
      4 -> Annotation(TOKEN, 4, 4, "学", Map()),
      6 -> Annotation(TOKEN, 6, 6, "生", Map()),
      9 -> Annotation(TOKEN, 8, 9, "理学", Map()))

    val automaton = new AhoCorasickAutomaton(japaneseSampleAlphabet, entityPatterns)
    automaton.buildMatchingMachine()
    val actualOutput = automaton.searchWords(sentence, tokens)

    val expectedOutput = List(
      Annotation(CHUNK, 0, 2, "音楽数", Map("entity" -> "Test-Japanese", "sentence" -> "0")),
      Annotation(CHUNK, 6, 6, "生", Map("entity" -> "Test-Japanese", "sentence" -> "0")))
    assert(actualOutput == expectedOutput)
  }

  it should "build a matching machine for Arabic" in {
    val arabicSampleAlphabet = "مرحبانجوسخت"
    val entityPatterns = Array(EntityPattern("Test-Arabic", Seq("من", "مختبرات")))
    val text = "مرحبا من جون سنو مختبرات"
    val sentence = Sentence(text, 0, text.length, 0)
    val tokens = Map(
      4 -> Annotation(TOKEN, 0, 4, "مرحبا", Map()),
      7 -> Annotation(TOKEN, 6, 7, "من", Map()),
      11 -> Annotation(TOKEN, 9, 11, "جون", Map()),
      15 -> Annotation(TOKEN, 13, 15, "سنو", Map()),
      23 -> Annotation(TOKEN, 17, 23, "مختبرات", Map()))

    val automaton = new AhoCorasickAutomaton(arabicSampleAlphabet, entityPatterns)
    automaton.buildMatchingMachine()
    val actualOutput = automaton.searchWords(sentence, tokens)

    val expectedOutput = List(
      Annotation(CHUNK, 6, 7, "من", Map("entity" -> "Test-Arabic", "sentence" -> "0")),
      Annotation(CHUNK, 17, 23, "مختبرات", Map("entity" -> "Test-Arabic", "sentence" -> "0")))
    assert(actualOutput == expectedOutput)
  }

  it should "raise error when building an automaton with incomplete alphabet" in {
    val englishAlphabet = "abcd"
    val entityPatterns = Array(EntityPattern("Test", Seq("ab", "cd", "ce")))

    val automaton = new AhoCorasickAutomaton(englishAlphabet, entityPatterns)
    assertThrows[UnsupportedOperationException] {
      automaton.buildMatchingMachine()
    }
  }

  it should "raise error when searching a word with a character not found on alphabet" in {
    val englishAlphabet = "abcdefghijklmnopqrstuvwxyz"
    val entityPatterns = Array(EntityPattern("Test", Seq("hello", "hi")))
    val text = "Hello there"
    val sentence = Sentence(text, 0, text.length, 0)
    val tokens = Map(
      4 -> Annotation(TOKEN, 0, 4, "Hello", Map()),
      6 -> Annotation(TOKEN, 6, 10, "here", Map()))

    val automaton =
      new AhoCorasickAutomaton(englishAlphabet, entityPatterns, caseSensitive = true)
    automaton.buildMatchingMachine()

    val errorMessage = intercept[UnsupportedOperationException] {
      automaton.searchWords(sentence, tokens)
    }

    assert(errorMessage.getMessage == "Char H not found on alphabet. Please check alphabet")
  }

  it should "build a case sensitive matching machine" in {
    val englishAlphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGH"
    val entityPatterns = Array(EntityPattern("Test", Seq("hello", "hi", "Hello")))
    val text = "Hello there"
    val sentence = Sentence(text, 0, text.length, 0)
    val tokens = Map(
      4 -> Annotation(TOKEN, 0, 4, "Hello", Map()),
      6 -> Annotation(TOKEN, 6, 10, "here", Map()))

    val automaton =
      new AhoCorasickAutomaton(englishAlphabet, entityPatterns, caseSensitive = true)
    automaton.buildMatchingMachine()
    val actualOutput = automaton.searchWords(sentence, tokens)

    val expectedOutput =
      List(Annotation(CHUNK, 0, 4, "Hello", Map("entity" -> "Test", "sentence" -> "0")))
    assert(actualOutput == expectedOutput)
  }

  it should "search multi-tokens" in {
    val englishAlphabet =
      "abcdefghijklmnopqrstuvwxyz" + "abcdefghijklmnopqrstuvwxyz".toUpperCase()
    val entityPatterns = Array(
      EntityPattern(
        "PER",
        Seq("Jon", "John", "John Snow", "Jon Snow", "Snow", "Doctor John Snow")),
      EntityPattern("LOC", Seq("United Kingdom", "United", "Kingdom", "Winterfell")))
    val text = "Doctor John Snow lives in the United Kingdom"
    val sentence = Sentence(text, 0, text.length, 0)
    val tokens = Map(
      5 -> Annotation(TOKEN, 0, 5, "Doctor", Map()),
      10 -> Annotation(TOKEN, 7, 10, "John", Map()),
      15 -> Annotation(TOKEN, 12, 15, "Snow", Map()),
      21 -> Annotation(TOKEN, 17, 21, "lives", Map()),
      24 -> Annotation(TOKEN, 23, 24, "in", Map()),
      28 -> Annotation(TOKEN, 26, 28, "the", Map()),
      35 -> Annotation(TOKEN, 30, 35, "United", Map()),
      43 -> Annotation(TOKEN, 37, 43, "Kingdom", Map()))

    val automaton =
      new AhoCorasickAutomaton(englishAlphabet, entityPatterns, caseSensitive = true)
    automaton.buildMatchingMachine()

    val expectedOutput = Seq(
      Annotation(CHUNK, 0, 15, "Doctor John Snow", Map("entity" -> "PER", "sentence" -> "0")),
      Annotation(CHUNK, 30, 43, "United Kingdom", Map("entity" -> "LOC", "sentence" -> "0")))
    val actualOutput = automaton.searchWords(sentence, tokens)
    assert(actualOutput == expectedOutput)
  }

}
