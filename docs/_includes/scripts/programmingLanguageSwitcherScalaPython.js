'use strict';

function tabs({tabsWrapperSelector, tabsParentSelector, tabsSelector, tabsContentSelector, activeClass}) {
    //Tabs

    const tabsWrapper = document.querySelectorAll(tabsWrapperSelector);

    //Detecting all tabs
    tabsWrapper.forEach(tab => {
        const tabsParent = tab.querySelector(tabsParentSelector),
                tabsLi = tab.querySelectorAll(tabsSelector),
                tabsContent = tab.querySelectorAll(tabsContentSelector);

        //Hiding all tabs
        function hideTabsContent() {
            if(Array.from(tabsLi).length != 0) {
                tabsContent.forEach(item => {
                    item.style.display = 'none';
                }); 
            }
            
            if(Array.from(tabsLi).length != 0) {
                tabsLi.forEach(item => {
                    item.classList.remove(activeClass);
                }); 
            }
        }

        //Show active tabs
        function showTabContent(i = 0) {
            if(Array.from(tabsContent).length != 0) {
                tabsContent[i].style.display = "block";
            }
            if(Array.from(tabsLi).length != 0) {
                tabsLi[i].classList.add(activeClass);            
            }
        }

        //Changing the tabs
        if(tabsParent != null) {
            tabsParent.addEventListener('click', (event) => {
                const target = event.target;
    
                if(target && target.classList.contains(tabsSelector.slice(1))) {
                    tabsLi.forEach((item, i) => {
                        if(target == item) {
                            hideTabsContent();
                            try{showTabContent(i);}catch(e){}
                        }
                    });
                }
            });
        }        
        
        hideTabsContent();
        showTabContent();
    });
}

tabs({
    tabsWrapperSelector: '.tabs-model-aproach', 
    tabsParentSelector: '.tabs-model-aproach-head', 
    tabsSelector: '.tab-li-model-aproach', 
    tabsContentSelector: '.tabs-python-scala-box', 
    activeClass: 'tabheader_active'
});
tabs({
    tabsWrapperSelector: '.tabs-python-scala-box', 
    tabsParentSelector: '.tabs-python-scala-head', 
    tabsSelector: '.tab-python-scala-li', 
    tabsContentSelector: '.tabs-mfl-box', 
    activeClass: 'tabheader_active'
});
tabs({
    tabsWrapperSelector: '.tabs-mfl-box', 
    tabsParentSelector: '.tabs-mfl-head', 
    tabsSelector: '.tab-mfl-li', 
    tabsContentSelector: '.tab-mfl-content', 
    activeClass: 'tabheader_active'
});
tabs({
    tabsWrapperSelector: '.tabs-box', 
    tabsParentSelector: '.tabs-python-scala-head', 
    tabsSelector: '.tab-python-scala-li', 
    tabsContentSelector: '.tabs-box .highlighter-rouge', 
    activeClass: 'tabheader_active'
});
tabs({
    tabsWrapperSelector: '.tabs-box', 
    tabsParentSelector: '.tabs-model-aproach-head', 
    tabsSelector: '.tab-li-model-aproach', 
    tabsContentSelector: '.tabs-python-scala-box', 
    activeClass: 'tabheader_active'
});
tabs({
    tabsWrapperSelector: '.tabs-box', 
    tabsParentSelector: '.tabs-model-aproach-head', 
    tabsSelector: '.tab-li-model-aproach', 
    tabsContentSelector: '.tabs-box .highlighter-rouge', 
    activeClass: 'tabheader_active'
});