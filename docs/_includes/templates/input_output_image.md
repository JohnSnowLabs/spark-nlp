{% if include.input_image and include.output_image %}
<div class="input_output_wrapper">
  <div class="input_output_inner">
    <div class="input_output_title">Input image</div>
{% if include.input_image %}

<figure class="shadow input_output_img" markdown="1">

{{include.input_image}}

</figure>

{% endif %}

        </div>
        <div class="input_output_inner">
          <div class="input_output_title">Output image</div>
{% if include.output_image %}

<figure class="shadow input_output_img" markdown="1">

{{include.output_image}}

</figure>

{% endif %}
  </div>
  <!-- The Modal/Lightbox -->
<div id="overview"></div>
  <div id="myModal" class="input_output_modal shadow">
      <a href="#" class="close-modal"><i class="fas fa-times"></i></a>
      <img src="" alt="">
  </div>
</div>
{% endif %}

<script> 
try {
      const imgCont = document.querySelectorAll('.input_output_img img'),
            modelImg = document.querySelector('#myModal img'),
            close = document.querySelector('.close-modal'),
            overview = document.querySelector('#overview'),
            model = document.querySelector('#myModal');

      let currentSrc = '';

      imgCont.forEach(item => {
            item.addEventListener('click', (e) => {
                  currentSrc = item.src;
                  modelImg.src = currentSrc;
                  model.classList.add('open');
                  overview.classList.add('open');
            })
      })

      close.addEventListener('click', (e) => {
            e.preventDefault();
            model.classList.remove('open');
            overview.classList.remove('open');
      }) 
      overview.addEventListener('click', () => {
            model.classList.remove('open');
            overview.classList.remove('open');
      })

} catch(e){}
</script>