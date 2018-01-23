package com.johnsnowlabs.util

object Benchmark {
  def time[R](description: String)(block: => R): R = {
    val t0 = System.nanoTime()
    val result = block
    val t1 = System.nanoTime()
    println(description + ": " + ((t1 - t0)/1000000000.0) + "sec")
    result
  }
}
