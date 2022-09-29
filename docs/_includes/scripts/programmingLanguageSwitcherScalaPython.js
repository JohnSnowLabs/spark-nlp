/* 
jQuery(document).ready(function(){  
    $( ".scala-button" ).click(function() {
        $(this).closest( ".tabs-box" ).find(".scala-button").removeClass('code-selector-un-active').addClass( "code-selector-active" );        

        //remove  active class from all other buttons
        $(this).closest( ".tabs-box" ).find(".nlu-button").removeClass('code-selector-active').addClass('code-selector-un-active');
        $(this).closest( ".tabs-box" ).find(".python-button").removeClass('code-selector-active').addClass('code-selector-un-active');

        //toggle language snippets
        $(this).closest( ".tabs-box" ).find( ".language-scala" ).show();
        $(this).closest( ".tabs-box" ).find( ".language-python, .nlu-block" ).hide();
    });

    $( ".python-button" ).click(function() {
        //set current button to active class and remove unactive class
        $(this).closest( ".tabs-box" ).find(".python-button").removeClass('code-selector-un-active').addClass( "code-selector-active" ); 

        //remove  active class from all other buttons
        $(this).closest( ".tabs-box" ).find(".nlu-button").removeClass('code-selector-active').addClass('code-selector-un-active');
        $(this).closest( ".tabs-box" ).find(".scala-button").removeClass('code-selector-active').addClass('code-selector-un-active');


        //toggle language snippets
        $(this).closest( ".tabs-box" ).find( ".language-python" ).show();
        $(this).closest( ".tabs-box" ).find( ".nlu-block, .language-scala" ).hide();
    });

    $( ".nlu-button" ).click(function() {
        //set current button to active class and remove unactive class
        $(this).closest( ".tabs-box" ).find(".nlu-button").removeClass('code-selector-un-active').addClass( "code-selector-active" );        

        //remove  active class from all other buttons
        $(this).closest( ".tabs-box" ).find(".scala-button").removeClass('code-selector-active').addClass('code-selector-un-active');
        $(this).closest( ".tabs-box" ).find(".python-button").removeClass('code-selector-active').addClass('code-selector-un-active');

        //toggle language snippets        
        $(this).closest( ".tabs-box" ).find( ".language-python, .language-scala" ).hide();
        $(this).closest( ".tabs-box" ).find( ".nlu-block" ).show();
    });
}); */

/* function togglePython1() {

    //set current button to active class and remove unactive class
    $( ".python-button" ).addClass( "code-selector-active" );


    //toggle language snippets
    $( ".tabs-box .language-python" ).show();
    $( ".tabs-box .nlu-block" ).hide();
    $( ".tabs-box .language-scala" ).hide();
}

function defer(method) { //wait until jquery ready
    if (window.jQuery) {
        method();
    } else {
        setTimeout(function() { defer(method); }, 15);
    }
}

defer(function () { // load inital language
    togglePython1();
}); */



//Second tabs
if(document.querySelectorAll('.tab-li-second')) {
    let tabLi = document.querySelectorAll('.tab-li-second');

    tabLi.forEach(element => {
        element.addEventListener('click', function(e) {
            e.preventDefault();
            let tabAttribute = element.getAttribute('data-type'),
                tabLiInner = element.parentNode.querySelectorAll('.tab-li-second'),
                tabBoxInner = element.parentNode.parentNode.parentNode.querySelectorAll('.tabs-box-medic-inner');
            
            //remove active class
            tabBoxInner.forEach(item => {
                item.classList.remove('active');
            });
            tabLiInner.forEach(el => {
                el.classList.remove('active');
            });
            element.classList.add('active');
    
            //add active class
            switch (tabAttribute) {
                case "python":
                    tabBoxInner.forEach(item => {
                        if(item.classList.contains('language-python')) {
                            item.classList.add('active');
                        }                    
                    });
                    break;
                case "scala":
                    tabBoxInner.forEach(item => {
                        if(item.classList.contains('language-scala')) {
                            item.classList.add('active');
                        }                    
                    });
                  break;
                default:              
              }
        });
    });
}

//Third tabs
if(document.querySelectorAll('.tab-li-inner')) {



    let tabLiSecond = document.querySelectorAll('.tab-li-inner'),
        tabLiTop = document.querySelectorAll('.toptab-second'),
        tabLi = document.querySelectorAll('.toptab-second p');

        console.log(tabLiTop);

    tabLiTop.forEach(e => {        
        e.nextElementSibling.classList.add('active');
    });

    tabLi.forEach(e => {        
        e.firstChild.classList.add('active');
    });


    tabLiSecond.forEach(element => {
        element.addEventListener('click', function(e) {
            e.preventDefault();
            let tabAttributeSecond = element.getAttribute('data-type'),
                tabLiInnerSecond = element.parentNode.querySelectorAll('.tab-li-inner'),
                tabBoxInnerSecond = element.parentNode.parentNode.parentNode.querySelectorAll('.tabs-box-medic-inner-second');
            
            //remove active class
            tabBoxInnerSecond.forEach(item => {
                item.classList.remove('active');
            });
            tabLiInnerSecond.forEach(el => {
                el.classList.remove('active');
            });
            element.classList.add('active');
    
            //add active class
            switch (tabAttributeSecond) {
                case "medical":
                    tabBoxInnerSecond.forEach(item => {
                        if(item.classList.contains('language-medical')) {
                            item.classList.add('active');
                        }                    
                    });
                    break;
                case "finance":
                    tabBoxInnerSecond.forEach(item => {
                        if(item.classList.contains('language-finance')) {
                            item.classList.add('active');
                        }                    
                    });
                  break;
                case "legal":
                tabBoxInnerSecond.forEach(item => {
                    if(item.classList.contains('language-legal')) {
                        item.classList.add('active');
                    }                    
                });
                break;
                default:              
              }
        });
    });
}