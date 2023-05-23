from pyspark import SparkContext
from pyspark.ml.param import TypeConverters
from pyspark.ml.wrapper import JavaWrapper


def base_type_to_java_class(spark_context, value_type):
    jvm = spark_context._gateway.jvm
    if value_type == bool:
        return jvm.java.lang.Boolean
    elif value_type == str:
        return jvm.java.lang.String
    elif value_type == int:
        return jvm.java.lang.Integer
    elif value_type == float:
        return jvm.java.lang.Double
    else:
        raise ValueError(
            (
                f"Tried to convert type for java array creation, but conversion from type {value_type}"
                f" to equivalent Java Class not implemented."
            )
        )


def type_converter_to_type_string(type_converter):
    if type_converter == TypeConverters.toListInt:
        return "List[int]"
    elif type_converter == TypeConverters.toList:
        return "List"
    elif type_converter == TypeConverters.toListFloat:
        return "List[float]"
    elif type_converter == TypeConverters.toListString:
        return "List[str]"
    elif type_converter == TypeConverters.toVector:
        return "Vector"
    elif type_converter == TypeConverters.toMatrix:
        return "Matrix"
    elif type_converter == TypeConverters.toFloat:
        return "float"
    elif type_converter == TypeConverters.toInt:
        return "int"
    elif type_converter == TypeConverters.toString:
        return "str"
    elif type_converter == TypeConverters.toBoolean:
        return "bool"
    else:
        return None


def list_elem_type(type_converter):
    """Tries to retrieve to type of list elements from the provided type converter.

    The type is required to create Java side arrays for the setters.

    Parameters
    ----------
    type_converter
        One of the functions of TypeConverters

    Returns
    -------
        The type of the elements of a list
    """

    if type_converter == TypeConverters.toListInt:
        return int
    elif type_converter == TypeConverters.toListFloat:
        return float
    elif type_converter == TypeConverters.toListString:
        return str
    else:
        return None


def list_to_java_array(value, type_converter):
    type_converter_elem_type = list_elem_type(type_converter)
    if len(value) > 0:
        elem_type = type(value[0])
        if (
                type_converter_elem_type is not None
                and elem_type != type_converter_elem_type
        ):
            print(
                "Warning: Conversion received a list with type that does not correspond the parameters type converter."
                " You might encounter errors when the java side function is called with this type."
            )
    else:
        elem_type = type_converter_elem_type
    if elem_type is None:
        raise ValueError(
            (
                "Passed an empty list to a setter, but could not infer the type of the elements."
                " The parameter might not have a TypeConverter defined."
                " Therefore, it's not possible to create java arrays to be passed to the JVM side setters."
                " If you think this setter should accept empty lists, please open an issue on GitHub."
            )
        )
    type_java_class = base_type_to_java_class(
        SparkContext._active_spark_context, elem_type
    )
    value_array = JavaWrapper._new_java_array(value, type_java_class)
    return value_array
