# Coloring the print
FAIL = '\033[91m'
ENDC = '\033[0m'


def check_obj_exist(java_class):
    # only call this after JVM is turned on
    from pyspark import SparkContext
    from pyspark.ml.util import _jvm
    from py4j.java_gateway import UserHelpAutoCompletion
    sc = SparkContext._active_spark_context
    java_obj = _jvm()
    for name in java_class.split("."):
        # Bug (?) in P4J. Even if ClassPath does not exist, JVM response is proto.SUCCESS_PACKAGE
        # But it should give Class Not Found in JMV Exception
        # Instead it gives confusing package not callable exception
        java_obj = getattr(java_obj, name)

    # If no class is loaded onto Java Obj, the Value of Dir is UserHelpAutoCompletion.KEY
    # see JavaPackage.__dir__ and JavaPackage.__apply__ methods for more infos
    if UserHelpAutoCompletion.KEY in dir(java_obj):
        bck = '\n'
        print(f"🚨 It looks like the class of {FAIL + java_class + ENDC} is missing from the SparkSession! 🚨\n This "
              f"usually means Spark NLP is missing from your SparkSession. Please make sure your {FAIL}spark.jar{ENDC}"
              f" is pointing to the correct Spark NLP Fat JAR, or {FAIL}spark.jar.packages{ENDC} is pointing to the "
              f"correct Maven coordinates. https://github.com/JohnSnowLabs/spark-nlp#usage")
        if sc.getConf().get("spark.jars"):
            print(f"Currently loaded Jars are: \n",
                  f'{bck.join(sc.getConf().get("spark.jars").split(";"))}')
        else:
            print(f'There are no custom JARs loaded into this Spark Session! '
                  f'Make sure either {FAIL}spark.jar{ENDC} or {FAIL}spark.jar.packages{ENDC} is correctly set in SparkSession!')
