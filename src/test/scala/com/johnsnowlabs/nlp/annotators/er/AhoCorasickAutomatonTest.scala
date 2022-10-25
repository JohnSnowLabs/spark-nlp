package com.johnsnowlabs.nlp.annotators.er

import com.johnsnowlabs.nlp.Annotation
import com.johnsnowlabs.nlp.AnnotatorType.CHUNK
import com.johnsnowlabs.nlp.annotators.common.Sentence
import com.johnsnowlabs.tags.FastTest
import org.scalatest.flatspec.AnyFlatSpec

class AhoCorasickAutomatonTest extends AnyFlatSpec {

  "AhoCorasickAutomatonV2Test" should "build a matching machine for english" taggedAs FastTest in {

    val keywords = Seq("he", "she", "his", "hers", "my babe", "her", "love")
    val entityPatterns = Array(EntityPattern("TEST", keywords))
    val englishAlphabet = EntityRulerUtil.loadAlphabet("english")
    val text = "Yady is my babe, and I love her."
    val sentence = Sentence(text, 0, text.length, 0)

    val automaton = new AhoCorasickAutomaton(englishAlphabet, entityPatterns)
    val actualOutput = automaton.searchPatternsInText(sentence)

    val expectedOutput = Seq(
      Annotation(CHUNK, 8, 14, "my babe", Map("entity" -> "TEST", "sentence" -> "0")),
      Annotation(CHUNK, 23, 26, "love", Map("entity" -> "TEST", "sentence" -> "0")),
      Annotation(CHUNK, 28, 30, "her", Map("entity" -> "TEST", "sentence" -> "0")))
    assert(actualOutput == expectedOutput)
  }

  it should "work with a custom alphabet" taggedAs FastTest in {
    val customAlphabet = "abcdefghijklmnopqrstuvwxyz,"
    val entityPatterns =
      Array(EntityPattern("Noun", Seq("he", "she")), EntityPattern("Pronoun", Seq("his", "hers")))
    val text = "she is over there, look his eyes met hers"
    val sentence = Sentence(text, 0, text.length, 0)

    val automaton = new AhoCorasickAutomaton(customAlphabet, entityPatterns)
    val actualOutput = automaton.searchPatternsInText(sentence)

    val expectedOutput = List(
      Annotation(CHUNK, 0, 2, "she", Map("entity" -> "Noun", "sentence" -> "0")),
      Annotation(CHUNK, 24, 26, "his", Map("entity" -> "Pronoun", "sentence" -> "0")),
      Annotation(CHUNK, 37, 40, "hers", Map("entity" -> "Pronoun", "sentence" -> "0")))
    assert(actualOutput == expectedOutput)
  }

  it should "build a matching machine for Japanese" taggedAs FastTest in {
    val japaneseSampleAlphabet = "ざ音楽数ポ学生理学ぎ"
    val entityPatterns = Array(EntityPattern("Test-Japanese", Seq("音楽数", "生")))
    val text = "音楽数 学 生 理学"
    val sentence = Sentence(text, 0, text.length, 0)

    val automaton = new AhoCorasickAutomaton(japaneseSampleAlphabet, entityPatterns)
    val actualOutput = automaton.searchPatternsInText(sentence)

    val expectedOutput = List(
      Annotation(CHUNK, 0, 2, "音楽数", Map("entity" -> "Test-Japanese", "sentence" -> "0")),
      Annotation(CHUNK, 6, 6, "生", Map("entity" -> "Test-Japanese", "sentence" -> "0")))
    assert(actualOutput == expectedOutput)
  }

  it should "build a matching machine for Arabic" taggedAs FastTest in {
    val arabicSampleAlphabet = "مرحبانجوسخت"
    val entityPatterns = Array(EntityPattern("Test-Arabic", Seq("من", "مختبرات")))
    val text = "مرحبا من جون سنو مختبرات"
    val sentence = Sentence(text, 0, text.length, 0)

    val automaton = new AhoCorasickAutomaton(arabicSampleAlphabet, entityPatterns)
    val actualOutput = automaton.searchPatternsInText(sentence)
    val expectedOutput = List(
      Annotation(CHUNK, 6, 7, "من", Map("entity" -> "Test-Arabic", "sentence" -> "0")),
      Annotation(CHUNK, 17, 23, "مختبرات", Map("entity" -> "Test-Arabic", "sentence" -> "0")))
    assert(actualOutput == expectedOutput)
  }

  it should "raise error when searching a word with a character not found on alphabet" taggedAs FastTest in {
    var englishAlphabet = "abcdefghijklmnopqrstuvwxyz"
    var entityPatterns = Array(EntityPattern("Test", Seq("hello", "hi")))
    var text = "Hello there"
    var sentence = Sentence(text, 0, text.length, 0)

    var automaton =
      new AhoCorasickAutomaton(englishAlphabet, entityPatterns, caseSensitive = true)

    var errorMessage = intercept[UnsupportedOperationException] {
      automaton.searchPatternsInText(sentence)
    }
    assert(errorMessage.getMessage == "Char H not found on alphabet. Please check alphabet")

    englishAlphabet = englishAlphabet + englishAlphabet.toUpperCase()
    entityPatterns = Array(EntityPattern("LOC", Seq("Gondor")))
    text = "Elendil used to live in Númenor"
    sentence = Sentence(text, 0, text.length, 0)
    automaton = new AhoCorasickAutomaton(englishAlphabet, entityPatterns, caseSensitive = true)
    errorMessage = intercept[UnsupportedOperationException] {
      automaton.searchPatternsInText(sentence)
    }
    assert(errorMessage.getMessage == "Char ú not found on alphabet. Please check alphabet")
  }

  it should "build a case sensitive matching machine" taggedAs FastTest in {
    val englishAlphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGH"
    val entityPatterns = Array(EntityPattern("Test", Seq("hello", "hi", "Hello")))
    val text = "Hello there"
    val sentence = Sentence(text, 0, text.length, 0)

    val automaton =
      new AhoCorasickAutomaton(englishAlphabet, entityPatterns, caseSensitive = true)
    val actualOutput = automaton.searchPatternsInText(sentence)

    val expectedOutput =
      List(Annotation(CHUNK, 0, 4, "Hello", Map("entity" -> "Test", "sentence" -> "0")))
    assert(actualOutput == expectedOutput)
  }

  it should "search multi-tokens" taggedAs FastTest in {
    val englishAlphabet =
      "abcdefghijklmnopqrstuvwxyz" + "abcdefghijklmnopqrstuvwxyz".toUpperCase()
    val entityPatterns = Array(
      EntityPattern(
        "PER",
        Seq("Jon", "John", "John Snow", "Jon Snow", "Snow", "Doctor John Snow")),
      EntityPattern("LOC", Seq("United Kingdom", "United", "Kingdom", "Winterfell")))
    val text = "Doctor John Snow lives in the United Kingdom"
    val sentence = Sentence(text, 0, text.length, 0)

    val automaton = new AhoCorasickAutomaton(englishAlphabet, entityPatterns)
    val actualOutput = automaton.searchPatternsInText(sentence)

    val expectedOutput = Seq(
      Annotation(CHUNK, 0, 15, "Doctor John Snow", Map("entity" -> "PER", "sentence" -> "0")),
      Annotation(CHUNK, 30, 43, "United Kingdom", Map("entity" -> "LOC", "sentence" -> "0")))
    assert(actualOutput == expectedOutput)

  }

}
