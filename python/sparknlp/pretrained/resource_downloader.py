class ResourceDownloader(object):

    @staticmethod
    def downloadModel(reader, name, language, remote_loc=None, j_dwn='PythonResourceDownloader'):
        print(name + " download started this may take some time.")
        file_size = _internal._GetResourceSize(name, language, remote_loc).apply()
        if file_size == "-1":
            print("Can not find the model to download please check the name!")
        else:
            print("Approximate size to download " + file_size)
            stop_threads = False
            t1 = threading.Thread(target=printProgress, args=(lambda: stop_threads,))
            t1.start()
            try:
                j_obj = _internal._DownloadModel(reader.name, name, language, remote_loc, j_dwn).apply()
            except Py4JJavaError as e:
                sys.stdout.write("\n" + str(e))
                raise e
            finally:
                stop_threads = True
                t1.join()

            return reader(classname=None, java_model=j_obj)
    @staticmethod
    def downloadModelDirectly(name, remote_loc="public/models"):
        _internal._DownloadModelDirectly(name, remote_loc).apply()

    @staticmethod
    def downloadPipeline(name, language, remote_loc=None):
        print(name + " download started this may take some time.")
        file_size = _internal._GetResourceSize(name, language, remote_loc).apply()
        if file_size == "-1":
            print("Can not find the model to download please check the name!")
        else:
            print("Approx size to download " + file_size)
            stop_threads = False
            t1 = threading.Thread(target=printProgress, args=(lambda: stop_threads,))
            t1.start()
            try:
                j_obj = _internal._DownloadPipeline(name, language, remote_loc).apply()
                jmodel = PipelineModel._from_java(j_obj)
            finally:
                stop_threads = True
                t1.join()

            return jmodel

    @staticmethod
    def clearCache(name, language, remote_loc=None):
        _internal._ClearCache(name, language, remote_loc).apply()

    @staticmethod
    def showPublicModels(annotator=None, lang=None, version=None):
        print(_internal._ShowPublicModels(annotator, lang, version).apply())

    @staticmethod
    def showPublicPipelines(lang=None, version=None):
        print(_internal._ShowPublicPipelines(lang, version).apply())

    @staticmethod
    def showUnCategorizedResources():
        print(_internal._ShowUnCategorizedResources().apply())

    @staticmethod
    def showAvailableAnnotators():
        print(_internal._ShowAvailableAnnotators().apply())

