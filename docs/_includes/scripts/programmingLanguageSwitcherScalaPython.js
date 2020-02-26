
function toggleScala() {
    //set current button to active class and remove unactive class
    $(".scala-button").removeClass('code-selector-un-active');
    $( ".scala-button" ).addClass( "code-selector-active" );


    //remove  active class from all other buttons
    $(".python-button").removeClass('code-selector-active');

    //set  unactive class from all other buttons
    $(".python-button").addClass('code-selector-un-active');

    //toggle language snippets
    $( ".language-scala" ).show() 
    $( ".language-python" ).hide()
    $( ".language-java" ).hide()

    }

function togglePython() {

    //set current button to active class and remove unactive class
    $(".python-button").removeClass('code-selector-un-active');
    $( ".python-button" ).addClass( "code-selector-active" );


    //remove  active class from all other buttons
    $(".scala-button").removeClass('code-selector-active');


    //set un active class from all other buttons
    $(".scala-button").addClass('code-selector-un-active');

    //toggle language snippets
    $( ".language-python" ).show() 
    $( ".language-java" ).hide()
    $( ".language-scala" ).hide()
    }

function defer(method) { //wait until jquery ready
    if (window.jQuery) {
        method();
    } else {
        setTimeout(function() { defer(method) }, 15);
    }
}

defer(function () { // load inital language
    toggleScala()
});