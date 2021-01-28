package com.johnsnowlabs.collections

import org.scalatest.FlatSpec
import com.johnsnowlabs.tags.{FastTest, SlowTest}


class SearchTrieSpec extends FlatSpec {
  val trie = SearchTrie(
    Array(
      Array("a", "b", "a", "b", "a"),
      Array("a", "a", "a")
    ), caseSensitive = false
  )

  val aTrie =  SearchTrie(
    Array(
      Array("a", "a", "a", "a", "a"),
      Array("a", "a", "a"),
      Array("a", "a"),
      Array("a")
    ), caseSensitive = false
  )

  val btrie = SearchTrie(
    Array(
      Array("a", "b", "a", "b"),
      Array("b", "a", "a")
    ), caseSensitive = false
  )


  "SearchTrie" should "create correct encode words" taggedAs FastTest in {
    assert(trie.vocabulary.size == 2)
    assert(trie.vocabulary("a") == 0)
    assert(trie.vocabulary("b") == 1)
  }

  "SearchTrie" should "create correct number of nodes" taggedAs FastTest in {
    assert(trie.nodes.size == 8)
  }

  "SearchTrie" should "correct fill nodes info" taggedAs FastTest in {
    val isLeaf = Seq(false, false, false, false, false, true, false, true)
    val trieIsLeaf = trie.nodes.map(n => n._2)
    assert(trieIsLeaf == isLeaf)

    val length = Seq(0, 1, 2, 3, 4, 5, 2, 3)
    val trieLength = trie.nodes.map(n => n._3)
    assert(trieLength == length)

    val pi = Seq(0, 0, 0, 1, 2, 3, 1, 6)
    val triePi = trie.nodes.map(n => n._1)
    assert(triePi == pi)
  }

  "SearchTrie" should "correct search" taggedAs FastTest in {
    val text = "a b a a a b a b a a a a".split(" ")
    val result = trie.search(text)

    assert(result == Seq((2, 4), (4, 8), (8, 10), (9, 11)))
  }

  "SearchTrie" should "correct handle out of vocabulary words" taggedAs FastTest in {
    val text = "a b a c a b a b a c a a a".split(" ")
    val result = trie.search(text)

    assert(result == Seq((4, 8), (10, 12)))
  }

  "SearchTrie" should "correctly calculate lastLeaf" taggedAs FastTest in {
    val lastLeafs = aTrie.nodes.map(n => n._4)
    val expected = Seq(-1, -1, 1, 2, 3, 3)

    assert(lastLeafs == expected)
  }

  "SearchTrie" should "correctly find substrings" taggedAs FastTest in {
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

  "SearchTrie" should "correct process something adding nodes for pi in different branch" taggedAs FastTest in {
    assert(btrie.nodes.size == 8)
    val pi = (0 until 8).map(i => btrie.pi(i)).toList
    assert(pi == List(0, 0, 5, 6, 2, 0, 1, 1))
  }
}