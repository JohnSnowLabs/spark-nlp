# Windows examples

* Few resolved issues which were faced while developing **CustomForNerDLPipeline** solution:
>> Known Issues
1. If anyone encounters an error like 'Could not find a suitable tensorflow graph'
    * Please find the solution at https://nlp.johnsnowlabs.com/docs/en/graph
2. Encountered an issue while working with play framework, which has a transitive dependency of guava jar
    * Solution: Use dependencyOverrides += "com.google.guava" % "guava" % "15.0"
    * Exception looks like below:
    > > Exception in thread "main" java.lang.IllegalAccessError: tried to access method com.google.common.base.Stopwatch.<init>()V from class org.apache.hadoop.mapred.FileInputFormat
            at org.apache.hadoop.mapred.FileInputFormat.getSplits(FileInputFormat.java:312)
            at org.apache.spark.rdd.HadoopRDD.getPartitions(HadoopRDD.scala:204).....
