class AnnotatorTransformer(JavaTransformer, AnnotatorJavaMLReadable, JavaMLWritable, ParamsGettersSetters):
    @keyword_only
    def __init__(self, classname):
        super(AnnotatorTransformer, self).__init__()
        kwargs = self._input_kwargs
        if 'classname' in kwargs:
            kwargs.pop('classname')
        self.setParams(**kwargs)
        self.__class__._java_class_name = classname
        self._java_obj = self._new_java_obj(classname, self.uid)

