package com.johnsnowlabs.collections

import org.scalatest.FlatSpec

class SearchTrieSpec extends FlatSpec {
  val wikiTrie = SearchTrie(
    Array(
      Array("a"),
      Array("a", "b"),
      Array("b", "a", "b"),
      Array("b", "c"),
      Array("b", "c", "a"),
      Array("c"),
      Array("c", "a", "a")
    )
  )

  val aTrie =  SearchTrie(
    Array(
      Array("a", "a", "a", "a", "a"),
      Array("a", "a", "a"),
      Array("a", "a"),
      Array("a")
    )
  )

  "SearchTrie" should "create a vocabulary from unique words in phrases" in {
    assert(wikiTrie.vocabulary.size == 3)
    assert(wikiTrie.vocabulary("a") == 0)
    assert(wikiTrie.vocabulary("b") == 1)
    assert(wikiTrie.vocabulary("c") == 2)
  }

  "SearchTrie" should "create correct number of nodes" in {
    assert(wikiTrie.nodes.size == 11)
  }

  "SearchTrie" should "correctly fill nodes info" in {
    val isLeaf = Seq(false, true, true, false, false, true, true, true, true, false, true)
    val trieIsLeaf = wikiTrie.nodes.map(n => n._2)
    assert(trieIsLeaf == isLeaf)

    val length = Seq(0, 1, 2, 1, 2, 3, 2, 3, 1, 2, 3)
    val trieLength = wikiTrie.nodes.map(n => n._3)
    assert(trieLength == length)

    val pi = Seq(0, 0, 3, 0, 1, 2, 8, 9, 0, 1, 1)
    val triePi = wikiTrie.nodes.map(n => n._1)
    assert(triePi == pi)
  }

  "SearchTrie" should "search correctly" in {
    val text = "a a b a b c a".split(" ")
    val result = wikiTrie.search(text)

    val expectedResult = Seq(
      (0, 0), (1, 1), (1, 2), (3, 3), (2, 4), (3, 4), (4, 5), (5, 5), (4, 6), (6, 6))

    assert(result == expectedResult)
  }

  "SearchTrie" should "correctly handle out of vocabulary words" in {
    val text = "d b e b a c b".split(" ")
    val result = wikiTrie.search(text)

    assert(result == Seq((4, 4), (5, 5)))
  }

  "SearchTrie" should "correctly calculate lastLeaf" in {
    val lastLeafs = wikiTrie.nodes.map(n => n._4)
    val expected = Seq(-1, -1, -1, -1, 1, 2, 8, 1, -1, 1, 1)

    assert(lastLeafs == expected)
  }

  "SearchTrie" should "correctly find substrings" in {
    val text = "a a a a c a a a a a a".split(" ")
    val result = aTrie.search(text)
    val shouldFound =
      Seq((0, 0), (1, 1), (2, 2), (3, 3), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10)) ++
        Seq((0, 1), (1, 2), (2, 3), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10)) ++
        Seq((0, 2), (1, 3), (5, 7), (6, 8), (7, 9), (8, 10)) ++
        Seq((5, 9), (6, 10))

    for (pair <- shouldFound) {
      assert(result.contains(pair))
    }

    assert(result.size == shouldFound.size)
  }
}