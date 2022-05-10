class RecursiveAnnotatorApproach(_internal.RecursiveEstimator, JavaMLWritable, _internal.AnnotatorJavaMLReadable,
                                 AnnotatorProperties,
                                 _internal.ParamsGettersSetters):
    @keyword_only
    def __init__(self, classname):
        _internal.ParamsGettersSetters.__init__(self)
        self.__class__._java_class_name = classname
        self._java_obj = self._new_java_obj(classname, self.uid)
        self._setDefault(lazyAnnotator=False)

    def _create_model(self, java_model):
        raise NotImplementedError('Please implement _create_model in %s' % self)

