package com.johnsnowlabs.nlp.annotators.tokenizer.moses

import com.johnsnowlabs.tags.FastTest
import org.scalatest.FlatSpec

class MosesTokenizerTest extends FlatSpec {
  val moses = new MosesTokenizer("en")

  "MosesTokenizer" should "encode words correclty" taggedAs FastTest in {
    val text = "This, is a sentence with weird\xbb symbols\u2026 appearing everywhere\xbf"
    val expected = "This , is a sentence with weird \\xbb symbols \\u2026 appearing everywhere \\xbf"
    val tokenized = moses.tokenize(text).mkString(" ")
    assert(tokenized == expected)
  }
  //   def test_moses_tokenize(self):
  //        moses = MosesTokenizer()
  //
  //        # Tokenize a sentence.
  //        text = (
  //            u"This, is a sentence with weird\xbb symbols\u2026 appearing everywhere\xbf"
  //        )
  //        expected_tokens = u"This , is a sentence with weird \xbb symbols \u2026 appearing everywhere \xbf"
  //        tokenized_text = moses.tokenize(text, return_str=True)
  //        assert tokenized_text == expected_tokens
  //
  //        # The nonbreaking prefixes should tokenize the final fullstop.
  //        assert moses.tokenize("abc def.") == [u"abc", u"def", u"."]
  //
  //        # The nonbreaking prefixes should deal the situation when numeric only prefix is the last token.
  //        # In below example, "pp" is the last element, and there is no digit after it.
  //        assert moses.tokenize("2016, pp.") == [u"2016", u",", u"pp", u"."]
  //
  //        # Test escape_xml
  //        text = "This ain't funny. It's actually hillarious, yet double Ls. | [] < > [ ] & You're gonna shake it off? Don't?"
  //        expected_tokens_with_xmlescape = [
  //            "This",
  //            "ain",
  //            "&apos;t",
  //            "funny",
  //            ".",
  //            "It",
  //            "&apos;s",
  //            "actually",
  //            "hillarious",
  //            ",",
  //            "yet",
  //            "double",
  //            "Ls",
  //            ".",
  //            "&#124;",
  //            "&#91;",
  //            "&#93;",
  //            "&lt;",
  //            "&gt;",
  //            "&#91;",
  //            "&#93;",
  //            "&amp;",
  //            "You",
  //            "&apos;re",
  //            "gonna",
  //            "shake",
  //            "it",
  //            "off",
  //            "?",
  //            "Don",
  //            "&apos;t",
  //            "?",
  //        ]
  //        expected_tokens_wo_xmlescape = [
  //            "This",
  //            "ain",
  //            "'t",
  //            "funny",
  //            ".",
  //            "It",
  //            "'s",
  //            "actually",
  //            "hillarious",
  //            ",",
  //            "yet",
  //            "double",
  //            "Ls",
  //            ".",
  //            "|",
  //            "[",
  //            "]",
  //            "<",
  //            ">",
  //            "[",
  //            "]",
  //            "&",
  //            "You",
  //            "'re",
  //            "gonna",
  //            "shake",
  //            "it",
  //            "off",
  //            "?",
  //            "Don",
  //            "'t",
  //            "?",
  //        ]
  //        assert moses.tokenize(text, escape=True) == expected_tokens_with_xmlescape
  //        assert moses.tokenize(text, escape=False) == expected_tokens_wo_xmlescape
  //
  //        # Test to check https://github.com/alvations/sacremoses/issues/19
  //        text = "this 'is' the thing"
  //        expected_tokens = ["this", "&apos;", "is", "&apos;", "the", "thing"]
  //        assert moses.tokenize(text, escape=True) == expected_tokens
}
