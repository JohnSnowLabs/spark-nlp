package com.johnsnowlabs.nlp.annotators.pos.perceptron

import org.apache.spark.util.AccumulatorV2

import scala.collection.mutable.{Map => MMap}

class TupleKeyDoubleMapAccumulator(defaultMap: MMap[(String, String), Double] = MMap.empty[(String, String), Double])
  extends AccumulatorV2[((String, String), Double), Map[(String, String), Double]] {

  @volatile private var mmap = defaultMap

  override def reset(): Unit = mmap.clear()

  override def add(v: ((String, String), Double)): Unit = mmap(v._1) += v._2

  def updateMany(other: MMap[(String, String), Double]): Unit = {
    println("ADDING MANY ON DOUBLE KEY")
    synchronized {
      other.foreach{case (k, v) =>
        mmap(k) = mmap.getOrElse(k, 0.0) + v
      }
    }
  }

  override def value: Map[(String, String), Double] = mmap.toMap

  override def copy(): AccumulatorV2[((String, String), Double), Map[(String, String), Double]] = {
    val c = new TupleKeyDoubleMapAccumulator(MMap.empty[(String, String), Double])
    c.mmap = this.mmap
    c
  }

  override def isZero: Boolean = mmap.isEmpty

  override def merge(other: AccumulatorV2[((String, String), Double), Map[(String, String), Double]]): Unit = {
    println("MERGE ON DOUBLE KEY")
    other match {
      case o: TupleKeyDoubleMapAccumulator =>
        updateMany(o.mmap)
      case _ => throw new Exception("Cannot merge tuple key long")
    }
  }
}

class TupleKeyLongMapAccumulator(defaultMap: MMap[(String, String), Long] = MMap.empty[(String, String), Long])
  extends AccumulatorV2[((String, String), Long), Map[(String, String), Long]] {

  @volatile var mmap = defaultMap

  override def reset(): Unit = mmap.clear()

  override def add(v: ((String, String), Long)): Unit = mmap(v._1) = v._2

  def updateMany(other: MMap[(String, String), Long]): Unit = {
    println("ADDING MANY ON LONG KEY")
    synchronized {
      mmap ++= other
    }
  }

  def update(k: (String, String), v: Long): Unit =  mmap(k) = v

  override def value: Map[(String, String), Long] = mmap.toMap

  override def copy(): AccumulatorV2[((String, String), Long), Map[(String, String), Long]] = {
    val c = new TupleKeyLongMapAccumulator(MMap.empty[(String, String), Long])
    c.mmap = this.mmap
    c
  }


  override def isZero: Boolean = mmap.isEmpty

  override def merge(other: AccumulatorV2[((String, String), Long), Map[(String, String), Long]]): Unit = {
    println("MERGE ON LONG KEY")
    other match {
      case o: TupleKeyLongMapAccumulator =>
        synchronized {
          o.mmap.foreach{case (k, v) =>
            mmap(k) = mmap.getOrElse(k, 0L) + v
          }
        }
      case _ => throw new Exception("Cannot merge tuple key long")
    }
  }
}

class StringMapStringDoubleAccumulator(defaultMap: MMap[String, MMap[String, Double]] = MMap.empty[String, MMap[String, Double]])
  extends AccumulatorV2[(String, MMap[String, Double]), Map[String, Map[String, Double]]] {

  @volatile private var mmap = defaultMap

  override def reset(): Unit = mmap.clear()

  override def add(v: (String, MMap[String, Double])): Unit = {
    mmap.update(v._1, mmap(v._1) ++ v._2)
  }

  override def value: Map[String, Map[String, Double]] = mmap.mapValues(_.toMap).toMap

  override def copy(): AccumulatorV2[(String, MMap[String, Double]), Map[String, Map[String, Double]]] = {
    val c = new StringMapStringDoubleAccumulator(MMap.empty[String, MMap[String, Double]])
    c.mmap ++= this.mmap
    c
  }

  override def isZero: Boolean = mmap.isEmpty

  def addMany(other: MMap[String, MMap[String, Double]]) = {
    other.foreach { case (k, v) => v.foreach { case (kk, vv) =>
      mmap(k) = mmap.getOrElse(k, MMap()) ++ MMap(kk -> vv)
    }}
  }

  override def merge(other: AccumulatorV2[(String, MMap[String, Double]), Map[String, Map[String, Double]]]): Unit =
    other match {
      case o: StringMapStringDoubleAccumulator =>
        o.mmap.foreach{case (k, v) =>
          v.foreach{case(kk,vv) =>
            mmap.getOrElseUpdate(k, MMap())(kk) = mmap(k).getOrElse(kk, 0.0) + vv
          }}
      case _ => throw new Exception("Wrong StringMapStringDouble merge")
    }
}