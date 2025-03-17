/*
 * Copyright 2017-2025 John Snow Labs
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.johnsnowlabs.nlp.annotators.cleaners.util

import com.johnsnowlabs.nlp.annotators.cleaners.util.CleanerHelper.{
  cleanBullets,
  cleanDashes,
  cleanExtraWhitespace,
  cleanNonAsciiChars,
  cleanOrderedBullets,
  cleanPostfix,
  cleanPrefix,
  cleanTrailingPunctuation,
  removePunctuation,
  replaceUnicodeCharacters
}
import com.johnsnowlabs.tags.FastTest
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.prop.TableDrivenPropertyChecks.forAll
import org.scalatest.prop.Tables.Table

class CleanerHelperTestSpec extends AnyFlatSpec {

  "cleanTrailingPunctuation" should "remove a trailing symbols" taggedAs FastTest in {
    val inputs = Seq("Hello.", "Hello,", "Hello:", "Hello;", "Hello,.", ";", "")
    val expectedOutputs = Seq("Hello", "Hello", "Hello", "Hello", "Hello")

    inputs.zip(expectedOutputs).foreach { case (input, expected) =>
      val actual = cleanTrailingPunctuation(input)
      assert(actual == expected)
    }
  }

  it should "not remove punctuation if none exists" taggedAs FastTest in {
    val inputs = Seq("Hello", "", "Hello, world!")
    val expectedOutputs = Seq("Hello", "", "Hello, world!")

    inputs.zip(expectedOutputs).foreach { case (input, expected) =>
      val actual = cleanTrailingPunctuation(input)
      assert(actual == expected)
    }
  }

  "cleanDashes" should "replace a single dash with a space" taggedAs FastTest in {
    val inputs = Seq(
      "Hello-World",
      "Hello---World",
      "Hello\u2013World",
      "Hello-World\u2013Scala",
      "-Hello World-",
      "---")
    val expectedOutputs =
      Seq("Hello World", "Hello   World", "Hello World", "Hello World Scala", "Hello World", "")

    inputs.zip(expectedOutputs).foreach { case (input, expected) =>
      val actual = cleanDashes(input)
      assert(actual == expected)
    }
  }

  it should "handle strings with no dashes without modifying them" taggedAs FastTest in {
    val inputs = Seq("Hello World", "")
    val expectedOutputs = Seq("Hello World", "")

    inputs.zip(expectedOutputs).foreach { case (input, expected) =>
      val actual = cleanDashes(input)
      assert(actual == expected)
    }
  }

  "cleanExtraWhitespace" should "replace non-breaking spaces with a single space" taggedAs FastTest in {
    val inputs = Seq(
      "Hello\u00a0World",
      "Hello\nWorld",
      "Hello   World",
      "Hello\u00a0\n  World",
      "  Hello World  ",
      "   ",
      "RISK\n\nFACTORS",
      "Item\\xa01A",
      "  Risk factors ",
      "Risk   factors ")
    val expectedOutputs = Seq(
      "Hello World",
      "Hello World",
      "Hello World",
      "Hello World",
      "Hello World",
      "",
      "RISK FACTORS",
      "Item 1A",
      "Risk factors",
      "Risk factors")

    inputs.zip(expectedOutputs).foreach { case (input, expected) =>
      val actual = cleanExtraWhitespace(input)
      assert(actual == expected)
    }
  }

  it should "handle strings with no whitespace without modifying them" taggedAs FastTest in {
    val inputs = Seq("HelloWorld", "")
    val expectedOutputs = Seq("HelloWorld", "")

    inputs.zip(expectedOutputs).foreach { case (input, expected) =>
      val actual = cleanExtraWhitespace(input)
      assert(actual == expected)
    }
  }

  "clean bullets" should "remove a leading bullet character" taggedAs FastTest in {
    val inputs = Seq(
      """● An excellent point!""",
      """●● An excellent point!""",
      """● An excellent point! ●●●""",
      """An excellent point!""",
      """Morse code! ●●●""")

    val expectedOutputs = Seq(
      "An excellent point!",
      """● An excellent point!""",
      "An excellent point! ●●●",
      "An excellent point!",
      "Morse code! ●●●")

    inputs.zip(expectedOutputs).foreach { case (input, expected) =>
      val actual = cleanBullets(input)
      assert(actual == expected)
    }
  }

  it should "remove a leading bullet unicode characters" taggedAs FastTest in {
    val inputs = Seq(
      "\u2022 Item 1",
      "\u2022  Item 2",
      "\u2043Item with dash bullet",
      "\u2022",
      "\u2022\u2022 Multiple bullets")

    val expectedOutputs =
      Seq("Item 1", "Item 2", "Item with dash bullet", "", "\u2022 Multiple bullets")

    inputs.zip(expectedOutputs).foreach { case (input, expected) =>
      val actual = cleanBullets(input)
      assert(actual == expected)
    }
  }

  it should "handle empty strings" in {
    val input = ""
    val expected = ""
    assert(cleanBullets(input) == expected)
  }

  it should "replace unicode characters" in {
    val inputs = Seq(
      """\x93A lovely quote!\x94""",
      """\x91A lovely quote!\x92""",
      """Our dog&apos;s bowl.""")
    val expectedOutputs = Seq("“A lovely quote!”", "‘A lovely quote!’", "Our dog's bowl.")

    inputs.zip(expectedOutputs).foreach { case (input, expected) =>
      assert(replaceUnicodeCharacters(input) == expected)
    }
  }

  it should "clean non-ascii characters" taggedAs FastTest in {
    val inputs = Seq(
      """\x88This text contains non-ascii characters!\x88""",
      """\x93A lovely quote!\x94""",
      """● An excellent point! ●●●""",
      """Item\xa01A""",
      """Our dog&apos;s bowl.""",
      """5 w=E2=80=99s""")

    val expectedOutputs = Seq(
      "This text contains non-ascii characters!",
      "A lovely quote!",
      " An excellent point! ",
      "Item1A",
      "Our dog's bowl.",
      "5 w=E2=80=99s")

    inputs.zip(expectedOutputs).foreach { case (input, expected) =>
      assert(cleanNonAsciiChars(input) == expected)
    }
  }

  "cleanOrderedBullets" should "remove ordered bullets" taggedAs FastTest in {
    val inputs = Seq(
      "1. Introduction:",
      "a. Introduction:",
      "20.3 Morse code ●●●",
      "5.3.1 Convolutional Networks ",
      "D.b.C Recurrent Neural Networks",
      "2.b.1 Recurrent Neural Networks",
      "eins. Neural Networks",
      "bb.c Feed Forward Neural Networks",
      "aaa.ccc Metrics",
      " version = 3.8",
      "1 2. 3 4",
      "1) 2. 3 4",
      "2,3. Morse code 3. ●●●",
      "1..2.3 four",
      "Fig. 2: The relationship",
      "23 is everywhere")

    val expectedOutputs = Seq(
      "Introduction:",
      "Introduction:",
      "Morse code ●●●",
      "Convolutional Networks",
      "Recurrent Neural Networks",
      "Recurrent Neural Networks",
      "eins. Neural Networks",
      "Feed Forward Neural Networks",
      "aaa.ccc Metrics",
      " version = 3.8",
      "1 2. 3 4",
      "1) 2. 3 4",
      "2,3. Morse code 3. ●●●",
      "1..2.3 four",
      "Fig. 2: The relationship",
      "23 is everywhere")

    inputs.zip(expectedOutputs).foreach { case (input, expected) =>
      assert(cleanOrderedBullets(input) == expected)
    }
  }

  "removePunctuation" should "remove punctuation" taggedAs FastTest in {
    val inputs = Seq("""“A lovely quote!”""", """‘A lovely quote!’""", """'()[]{};:'\",.?/\\-_""")

    val expectedOutputs = Seq("A lovely quote", "A lovely quote", "")

    inputs.zip(expectedOutputs).foreach { case (input, expected) =>
      val actual = removePunctuation(input)
      assert(actual == expected)
    }
  }

  "cleanPrefix" should "remove the prefix and any following punctuation/whitespace" taggedAs FastTest in {
    val testCases = Table(
      ("description", "text", "pattern", "ignoreCase", "strip", "expected"),
      (
        "Standard summary removal",
        "SUMMARY: A great SUMMARY",
        "(SUMMARY|DESC)",
        false,
        true,
        "A great SUMMARY"),
      (
        "Desc removal with case-sensitive match",
        "DESC: A great SUMMARY",
        "(SUMMARY|DESC)",
        false,
        true,
        "A great SUMMARY"),
      (
        "Without extra stripping",
        "SUMMARY: A great SUMMARY",
        "(SUMMARY|DESC)",
        false,
        false,
        "A great SUMMARY"),
      (
        "Removal with case ignored",
        "desc: A great SUMMARY",
        "(SUMMARY|DESC)",
        true,
        true,
        "A great SUMMARY"))

    forAll(testCases) { (desc, text, pattern, ignoreCase, strip, expected) =>
      withClue(s"Failed in case: $desc") {
        val actual = cleanPrefix(text, pattern, ignoreCase, strip)
        assert(actual == expected)
      }
    }
  }

  "cleanPostfix" should "remove the postfix and any following punctuation/whitespace" taggedAs FastTest in {
    val testCases = Table(
      ("description", "text", "pattern", "ignoreCase", "strip", "expected"),
      ("Remove trailing 'END' with strip", "The END! END", "(END|STOP)", false, true, "The END!"),
      (
        "Remove trailing 'STOP' with strip",
        "The END! STOP",
        "(END|STOP)",
        false,
        true,
        "The END!"),
      (
        "Keep trailing whitespace when not stripping",
        "The END! END",
        "(END|STOP)",
        false,
        false,
        "The END! "),
      (
        "Remove trailing 'end' ignoring case",
        "The END! end",
        "(END|STOP)",
        true,
        true,
        "The END!"))

    forAll(testCases) { (description, text, pattern, ignoreCase, strip, expected) =>
      withClue(s"Failed in case: $description") {
        val actual = cleanPostfix(text, pattern, ignoreCase, strip)
        assert(actual == expected)
      }
    }
  }

  "bytesStringToAnnotation" should "correctly decode a hex-encoded UTF-8 byte string containing Chinese characters" in {
    val text = """\xe6\xaf\x8f\xe6\x97\xa5\xe6\x96\xb0\xe9\x97\xbb"""
    val encoding = "utf-8"
    val expected = "每日新闻"

    val actual = CleanerHelper.bytesStringToString(text, encoding)

    assert(actual == expected)
  }

  it should "correctly decode a hex-encoded UTF-8 byte string containing emoticons" taggedAs FastTest in {
    val text = """Hello ð\x9f\x98\x80"""
    val encoding = "utf-8"
    val expected = "Hello 😀"

    val actual = CleanerHelper.bytesStringToString(text, encoding)

    assert(actual == expected)
  }

}
