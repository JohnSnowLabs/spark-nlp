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
function openTabCall(cityName){
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
}

/*OPen by URL*/
$(document).ready(function () {  
  const tabName = (window.location.hash || '').replace('#', '');
  const tab = document.getElementById(tabName || 'opensource');
  if (tab) {
    tab.click();
  }
});

jQuery(document).ready(function(){
	jQuery('.tab-item').click(function(event) {		
		if (($(window).width() > 400) && ($(window).width() < 1199))
	    {
	    	jQuery('.tab-item').removeClass('open');
	        jQuery(this).toggleClass('open');
	    }
  });
  

});


 