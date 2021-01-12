
jQuery(document).ready(function(){  
    $( ".scala-button" ).click(function() {
        $(this).closest( ".tabs-box" ).find(".scala-button").removeClass('code-selector-un-active').addClass( "code-selector-active" );
        

        //remove  active class from all other buttons
        $(this).closest( ".tabs-box" ).find(".java-button").removeClass('code-selector-active').addClass('code-selector-un-active');
        $(this).closest( ".tabs-box" ).find(".python-button").removeClass('code-selector-active').addClass('code-selector-un-active');

        //toggle language snippets
        $(this).closest( ".tabs-box" ).find( ".language-scala" ).show();
        $(this).closest( ".tabs-box" ).find( ".language-python, .language-java" ).hide();
    });

    $( ".python-button" ).click(function() {
        //set current button to active class and remove unactive class
        $(this).closest( ".tabs-box" ).find(".python-button").removeClass('code-selector-un-active').addClass( "code-selector-active" );        


        //remove  active class from all other buttons
        $(this).closest( ".tabs-box" ).find(".java-button").removeClass('code-selector-active').addClass('code-selector-un-active');
        $(this).closest( ".tabs-box" ).find(".scala-button").removeClass('code-selector-active').addClass('code-selector-un-active');


        //toggle language snippets
        $(this).closest( ".tabs-box" ).find( ".language-python" ).show();
        $(this).closest( ".tabs-box" ).find( ".language-java, .language-scala" ).hide();
    });

    $( ".java-button" ).click(function() {
        //set current button to active class and remove unactive class
        $(this).closest( ".tabs-box" ).find(".java-button").removeClass('code-selector-un-active').addClass( "code-selector-active" );        

        //remove  active class from all other buttons
        $(this).closest( ".tabs-box" ).find(".scala-button").removeClass('code-selector-active').addClass('code-selector-un-active');
        $(this).closest( ".tabs-box" ).find(".python-button").removeClass('code-selector-active').addClass('code-selector-un-active');

        //toggle language snippets
        $(this).closest( ".tabs-box" ).find( ".language-java" ).show();
        $(this).closest( ".tabs-box" ).find( ".language-python, .language-scala" ).hide();
    });
});

function togglePython1() {

    //set current button to active class and remove unactive class
    $( ".python-button" ).addClass( "code-selector-active" );


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
    togglePython1()
});


