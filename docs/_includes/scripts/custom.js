/* Responsive menu
	 ========================================================*/
jQuery(document).ready(function($) {
	jQuery('#responsive_menu').click(function(e) {
      e.preventDefault();
      jQuery(this).toggleClass('close');
      jQuery('.top_navigation').toggleClass('open');
  });
  jQuery('#aside_menu').click(function(e) {
      e.preventDefault();
      jQuery(this).toggleClass('close');
      jQuery('.js-col-aside').toggleClass('open');
      if (jQuery(window).width() <= 1023)
      {
        jQuery('.page__sidebar').toggleClass('open'); 
      jQuery('.demomenu').toggleClass('open');
      }
  });
  jQuery('.toc--ellipsis a').click(function(e) {
    if (jQuery(window).width() <= 767)
      {
        jQuery('.js-col-aside').removeClass('open');
        jQuery('.page__sidebar').removeClass('open');    
        jQuery('#aside_menu').removeClass('close');  
      }       
  });
});

/*TABS*/
/* function openTabCall(cityName){
  // Declare all variables
  var i, tabcontent, tablinks;

  // Get all elements with class="tabcontent" and hide them
  tabcontent = document.getElementsByClassName("tabcontent");
  for (i = 0; i < tabcontent.length; i++) {
    tabcontent[i].style.display = "none";
  }

  // Get all elements with class="tablinks" and remove the class "active"
  tablinks = document.getElementsByClassName("tablinks");
  for (i = 0; i < tablinks.length; i++) {
    tablinks[i].className = tablinks[i].className.replace(" active", "");
  }

  // Show the current tab, and add an "active" class to the button that opened the tab
  document.getElementById(cityName).style.display = "block";
}

function openTab(evt, cityName) {
  openTabCall(cityName);
  evt.currentTarget.className += " active";
} */

/*OPen by URL*/
$(document).ready(function () {  
  const tabName = (window.location.hash || '').replace('#', '');
  const tab = document.getElementById(tabName || 'opensource');
  if (tab) {
    tab.click();
  }
});

//Accordion demos categories
let acc = document.getElementsByClassName("acc-top"),
    isResizeble = false;

  if(!isResizeble) {
      let accBody = document.querySelector('.acc-body li.active');
      accBody.parentElement.style.maxHeight = accBody.parentElement.scrollHeight + 20 + "px";
      accBody.parentElement.classList.add('open');
      accBody.parentElement.previousElementSibling.classList.add('active');
      isResizeble = true;
  }

for (let i = 0; i < acc.length; i++) {
  acc[i].addEventListener("click", function() {
    this.classList.toggle("active");
    var panel = this.nextElementSibling;
    if (panel.style.maxHeight) {
      panel.style.maxHeight = null;
      panel.classList.remove('open');
    } else {
      panel.style.maxHeight = panel.scrollHeight + 20 + "px";
      panel.classList.add('open');
    }
  });
}

//Show more in demos description
let tabDescription = document.querySelectorAll('.tab-description');

tabDescription.forEach(element => {
  let tabDescriptionInner = element.querySelector('.tab-description-inner');
  if(element.offsetHeight < tabDescriptionInner.offsetHeight) {
    element.classList.add('big-descr');
  }
});

let showMore = document.querySelectorAll('.show_more');

showMore.forEach(element => {
  element.addEventListener("click", function(e) {
    e.preventDefault();
    this.parentElement.parentElement.classList.remove('big-descr');
    this.parentElement.parentElement.classList.add('big-descr-close');
  });
});