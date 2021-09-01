jQuery(document).ready(function () {
    $(".prediction-button").click(function () {
        $(this).closest(".tabs-box").find(".prediction-button").removeClass('code-selector-un-active').addClass("code-selector-active");

        // remove active class from all other buttons
        $(this).closest(".tabs-box").find(".training-button").removeClass('code-selector-active').addClass('code-selector-un-active');
        $(this).closest(".tabs-box").find(".embeddings-button").removeClass('code-selector-active').addClass('code-selector-un-active');

        //toggle content
        $(this.parentNode).siblings(".tabs-box.training-content").hide()
        $(this.parentNode).siblings(".tabs-box.embeddings-content").hide()
        $(this.parentNode).siblings(".tabs-box.prediction-content").show()
    });

    $(".training-button").click(function () {
        $(this).closest(".tabs-box").find(".training-button").removeClass('code-selector-un-active').addClass("code-selector-active");

        // remove active class from all other buttons
        $(this).closest(".tabs-box").find(".prediction-button").removeClass('code-selector-active').addClass('code-selector-un-active');
        $(this).closest(".tabs-box").find(".embeddings-button").removeClass('code-selector-active').addClass('code-selector-un-active');

        //toggle content
        $(this.parentNode).siblings(".tabs-box.prediction-content").hide()
        $(this.parentNode).siblings(".tabs-box.embeddings-content").hide()
        $(this.parentNode).siblings(".tabs-box.training-content").show()
    });

    $(".embeddings-button").click(function () {
        $(this).closest(".tabs-box").find(".embeddings-button").removeClass('code-selector-un-active').addClass("code-selector-active");

        // remove active class from all other buttons
        $(this).closest(".tabs-box").find(".training-button").removeClass('code-selector-active').addClass('code-selector-un-active');
        $(this).closest(".tabs-box").find(".prediction-button").removeClass('code-selector-active').addClass('code-selector-un-active');

        //toggle content
        $(this.parentNode).siblings(".tabs-box.training-content").hide()
        $(this.parentNode).siblings(".tabs-box.prediction-content").hide()
        $(this.parentNode).siblings(".tabs-box.embeddings-content").show()
    });
});
