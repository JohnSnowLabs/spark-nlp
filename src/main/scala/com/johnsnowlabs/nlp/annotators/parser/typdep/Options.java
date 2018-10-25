package com.johnsnowlabs.nlp.annotators.parser.typdep;

public class Options {

    String unimapFile = null;

    int numberOfPreTrainingIterations;
    int numberOfTrainingIterations;
    boolean initTensorWithPretrain;
    float regularization ;
    float gammaLabel;
    int rankFirstOrderTensor;
    int rankSecondOrderTensor;

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
