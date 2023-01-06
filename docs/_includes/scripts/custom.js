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

/*OPen by URL*/
jQuery(document).ready(function () {  
  const tabName = (window.location.hash || '').replace('#', '');
  const tab = document.getElementById(tabName || 'opensource');
  if (tab) {
    tab.click();
  }
});

//Accordion demos categories
if(document.querySelector(".acc-top")) {
  let acc = document.getElementsByClassName("acc-top"),
    isResizeble = false;

  if(!isResizeble && document.querySelector(".acc-top")) {
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
}


//Show more in demos description
if(document.querySelector('.tab-description')) {
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
}


//disable Colab link
if(document.querySelector('.btn.disable')) {
  let btnDisable = document.querySelectorAll('.btn.disable');

  btnDisable.forEach(element => {
    element.addEventListener("click", function(e) {
      e.preventDefault();
    });
  });
}


// Ancor click
const anchors = [].slice.call(document.querySelectorAll('.btn-box-install a')),
      animationTime = 300,
      framesCount = 20;

anchors.forEach(function(item) {
    item.addEventListener('click', function(e) {
        e.preventDefault();
        let coordY = document.querySelector(item.getAttribute('href')).getBoundingClientRect().top + window.pageYOffset -100;
    
        let scroller = setInterval(function() {
            let scrollBy = coordY / framesCount;
      
      if(scrollBy > window.pageYOffset - coordY && window.innerHeight + window.pageYOffset < document.body.offsetHeight) {
          window.scrollBy(0, scrollBy);
      } else {
                window.scrollTo(0, coordY);
        clearInterval(scroller);
      }
        }, animationTime / framesCount);
  });
}); 


//Pagination active
if(document.querySelector('.pagination_big')) {
  let paginationItems = document.querySelectorAll('.pagination_big li'),
      nextVersionContainer = document.querySelector('#nextver'),
      previosVersionContainer = document.querySelector('#previosver'),
      currentVersionContainer = document.querySelector('#currversion'),
      currentPageTitle = document.querySelector('#section').innerText;

  // Set active page and update version containers
  for (let i = 0; i < paginationItems.length; i++) {
    const item = paginationItems[i];
    const itemTitle = item.firstElementChild.innerHTML;
    if (itemTitle === currentPageTitle) {
      item.classList.add('active');
      currentVersionContainer.textContent = itemTitle;       
      if(item.previousElementSibling) {
        previosVersionContainer.textContent = item.previousElementSibling.innerText; 
        previosVersionContainer.parentElement.href += item.previousElementSibling.innerText.replaceAll('.', '_');
      } else {
        previosVersionContainer.parentElement.parentElement.classList.add('hide');
      }
      if(item.nextElementSibling) {
        nextVersionContainer.textContent = item.nextElementSibling.innerText;
        nextVersionContainer.parentElement.href += item.nextElementSibling.innerText.replaceAll('.', '_');
      } else {
        nextVersionContainer.parentElement.parentElement.classList.add('hide');
      }         
      break;
    }
  }
}