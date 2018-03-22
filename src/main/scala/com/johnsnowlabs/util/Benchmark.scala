package com.johnsnowlabs.util

object Benchmark {
  def time[R](description: String)(block: => R): R = {
    val t0 = System.nanoTime()
    val result = block
    val t1 = System.nanoTime()
    println(description + ": " + ((t1 - t0)/1000000000.0) + "sec")
    result
  }

  def measure(iterations: Integer = 3, msg: Boolean = true, description: String = "Took")(f: => Any): Double = {
    val time = (0 until iterations).map { _ =>
      val t0 = System.nanoTime()
      f
      System.nanoTime() - t0
    }.sum.toDouble / iterations

    if (msg)
      println(s"$description (Avg for $iterations iterations): ${time / 1000000000} sec")

    time
  }

  def measure(f: => Any): Double = measure()(f)
  def measure(d: String)(f: => Any): Double = measure(description = d)(f)
}