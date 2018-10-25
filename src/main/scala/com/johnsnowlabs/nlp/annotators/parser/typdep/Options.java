package com.johnsnowlabs.nlp.annotators.parser.typdep;

public class Options {

    String unimapFile = null;

    int numberOfPreTrainingIterations = 2;
    int numberOfTrainingIterations = 10;
    boolean initTensorWithPretrain = true;
    float regularization = 0.01f;
    float gammaLabel = 0;
    int rankFirstOrderTensor = 50;
    int rankSecondOrderTensor = 30;

    public Options() {

    }

    //TODO remove this attribute, the model should be saved in TypedDepdencyApproach
    public String modelFile = "example.model";

    private Options(Options options) {
        this.numberOfPreTrainingIterations = options.numberOfPreTrainingIterations;
        this.numberOfTrainingIterations = options.numberOfTrainingIterations;
        this.initTensorWithPretrain = options.initTensorWithPretrain;
        this.regularization = options.regularization;
        this.gammaLabel = options.gammaLabel;
        this.rankFirstOrderTensor = options.rankFirstOrderTensor;
        this.rankSecondOrderTensor = options.rankSecondOrderTensor;
    }

    static Options newInstance(Options options){
        //Copy factory
        return new Options(options);
    }

}
