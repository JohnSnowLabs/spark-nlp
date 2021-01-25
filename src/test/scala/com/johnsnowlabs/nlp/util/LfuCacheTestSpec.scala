package com.johnsnowlabs.nlp.util

import com.johnsnowlabs.tags.FastTest
import org.scalatest._

/** Inspired on
  * Lucas Torri
  * https://gist.github.com/lucastorri/138acfdd5f9903e5cf8e3cc1e7cbb8e7
  * */
class LfuCacheTestSpec extends FlatSpec {

  "A LfuCache" should "automatically adjust to new content" taggedAs FastTest in {

    val size = 10
    val lfu = new LfuCache[Int, Int](size)
    lfu.get(3)


    lfu.getOrElseUpdate(0, 5)
    lfu.get(5)

    lfu.getOrElseUpdate(1, 7)
    lfu.get(7)
    lfu.get(7)
    lfu.get(7)
    lfu.get(7)

    lfu.removeLast()
    lfu.removeLast()
    lfu.removeLast()

    (0 to size).foreach(i => lfu.getOrElseUpdate(i, i))
    (0 to size).foreach(lfu.get)
    (0 to size).foreach(_ => lfu.removeLast())

    (0 to size).foreach { i =>
      lfu.getOrElseUpdate(i, i)
      (0 until i).foreach(_ => lfu.get(i))
    }

  }

  "A LfuCache" should "Evict correctly" taggedAs FastTest in {

    val lfu = new LfuCache[String, Double](5)

    val iv = Seq(
      ("a", 234.5),
      ("b", 345.6),
      ("c", 456.7),
      ("d", 567.8),
      ("e", 678.9)
    )

    iv.foreach{case (i, v) => lfu.getOrElseUpdate(i, v)}

    assert(lfu.getSize == 5, "Wrong initial size")

    lfu.getOrElseUpdate("j", 25.1)
    lfu.get("b").getOrElse(lfu.getOrElseUpdate("b", 345.6))
    lfu.get("a").getOrElse(lfu.getOrElseUpdate("a", 234.5))
    lfu.get("a").getOrElse(lfu.getOrElseUpdate("a", 234.5))
    lfu.get("d").getOrElse(lfu.getOrElseUpdate("d", 567.8))
    lfu.getOrElseUpdate("r", 22.7)
    lfu.get("e").getOrElse(lfu.getOrElseUpdate("e", 678.9))
    lfu.get("e").getOrElse(lfu.getOrElseUpdate("e", 678.9))
    lfu.get("b").getOrElse(lfu.getOrElseUpdate("b", 345.6))

    assert(lfu.getSize == 5, "Size not as expected after getting and updated cache")

    lfu.getOrElseUpdate("even newer", 4.13)

    assert(lfu.getSize == 5, "Size not as expected after adding 2 new values")
    assert(lfu.get("even newer").isDefined, "Recently added key is not in the lfu!")

    assert(lfu.get("c").isEmpty, "value 'c' should not be in lfu since it was never used")
    assert(lfu.get("r").isEmpty, "value 'd' should not be in lfu since it was rarely queried (once)")
  }

}
