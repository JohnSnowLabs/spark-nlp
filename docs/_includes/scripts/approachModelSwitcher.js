/* jQuery(document).ready(function () {
    $(".model-button").click(function () {
        $(this).closest(".tabs-box").find(".model-button").removeClass('code-selector-un-active').addClass("code-selector-active");

        //remove  active class from all other buttons
        $(this).closest(".tabs-box").find(".approach-button").removeClass('code-selector-active').addClass('code-selector-un-active');

        //toggle content
        $(this.parentNode).siblings(".h3-box.approach-content").hide()
        $(this.parentNode).siblings(".h3-box.model-content").show()
    });

    $(".approach-button").click(function () {
        //set current button to active class and remove unactive class
        $(this).closest(".tabs-box").find(".approach-button").removeClass('code-selector-un-active').addClass("code-selector-active");

        //remove  active class from all other buttons
        $(this).closest(".tabs-box").find(".model-button").removeClass('code-selector-active').addClass('code-selector-un-active');

        //toggle content
        $(this.parentNode).siblings(".h3-box.model-content").hide()
        $(this.parentNode).siblings(".h3-box.approach-content").show()
    });
});
 */