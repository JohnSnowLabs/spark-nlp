class ReadAs(object):
    """Object that contains constants for how to read Spark Resources.

    Possible values are:

    ================= =======================================
    Value             Description
    ================= =======================================
    ``ReadAs.TEXT``   Read the resource as text.
    ``ReadAs.SPARK``  Read the resource as a Spark DataFrame.
    ``ReadAs.BINARY`` Read the resource as a binary file.
    ================= =======================================
    """
    TEXT = "TEXT"
    SPARK = "SPARK"
    BINARY = "BINARY"

