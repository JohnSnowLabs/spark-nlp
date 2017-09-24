###
Lightbox for Bootstrap 3 by @ashleydw
https://github.com/ashleydw/lightbox

License: https://github.com/ashleydw/lightbox/blob/master/LICENSE
###
"use strict";

$ = jQuery

EkkoLightbox = ( element, options ) ->

	@options = $.extend({
		title : null
		footer : null
		remote : null
	}, $.fn.ekkoLightbox.defaults, options || {})

	@$element = $(element)
	content = ''

	@modal_id = if @options.modal_id then @options.modal_id else 'ekkoLightbox-' + Math.floor((Math.random() * 1000) + 1)
	header = '<div class="modal-header"'+(if @options.title or @options.always_show_close then '' else ' style="display:none"')+'><button type="button" class="close" data-dismiss="modal" aria-hidden="true">&times;</button><h4 class="modal-title">' + (@options.title || "&nbsp;") + '</h4></div>'
	footer = '<div class="modal-footer"'+(if @options.footer then '' else ' style="display:none"')+'>' + @options.footer + '</div>'
	$(document.body).append '<div id="' + @modal_id + '" class="ekko-lightbox modal fade" tabindex="-1"><div class="modal-dialog"><div class="modal-content">' + header + '<div class="modal-body"><div class="ekko-lightbox-container"><div></div></div></div>' + footer + '</div></div></div>'

	@modal = $ '#' + @modal_id
	@modal_dialog = @modal.find('.modal-dialog').first()
	@modal_content = @modal.find('.modal-content').first()
	@modal_body = @modal.find('.modal-body').first()

	@lightbox_container = @modal_body.find('.ekko-lightbox-container').first()
	@lightbox_body = @lightbox_container.find('> div:first-child').first()

	@showLoading()
	@modal_arrows = null

	@border = {
		top: parseFloat(@modal_dialog.css('border-top-width')) + parseFloat(@modal_content.css('border-top-width')) + parseFloat(@modal_body.css('border-top-width'))
		right: parseFloat(@modal_dialog.css('border-right-width')) + parseFloat(@modal_content.css('border-right-width')) + parseFloat(@modal_body.css('border-right-width'))
		bottom: parseFloat(@modal_dialog.css('border-bottom-width')) + parseFloat(@modal_content.css('border-bottom-width')) + parseFloat(@modal_body.css('border-bottom-width'))
		left: parseFloat(@modal_dialog.css('border-left-width')) + parseFloat(@modal_content.css('border-left-width')) + parseFloat(@modal_body.css('border-left-width'))
	}

	@padding = {
		top: parseFloat(@modal_dialog.css('padding-top')) + parseFloat(@modal_content.css('padding-top')) + parseFloat(@modal_body.css('padding-top'))
		right: parseFloat(@modal_dialog.css('padding-right')) + parseFloat(@modal_content.css('padding-right')) + parseFloat(@modal_body.css('padding-right'))
		bottom: parseFloat(@modal_dialog.css('padding-bottom')) + parseFloat(@modal_content.css('padding-bottom')) + parseFloat(@modal_body.css('padding-bottom'))
		left: parseFloat(@modal_dialog.css('padding-left')) + parseFloat(@modal_content.css('padding-left')) + parseFloat(@modal_body.css('padding-left'))
	}

	@modal
	.on('show.bs.modal', @options.onShow.bind(@))
	.on 'shown.bs.modal', =>
		@modal_shown()
		@options.onShown.call(@)
	.on('hide.bs.modal', @options.onHide.bind(@))
	.on 'hidden.bs.modal', =>
		if @gallery
			$(document).off 'keydown.ekkoLightbox'
		@modal.remove()
		@options.onHidden.call(@)
	.modal 'show', options

	@modal

EkkoLightbox.prototype = {
	modal_shown: ->
		# when the modal first loads
		if !@options.remote
			@error 'No remote target given'
		else

			@gallery = @$element.data('gallery')
			if @gallery
				# parents('document.body') fails for some reason, so do this manually
				if this.options.gallery_parent_selector == 'document.body' || this.options.gallery_parent_selector == ''
					@gallery_items = $(document.body).find('*[data-toggle="lightbox"][data-gallery="' + @gallery + '"]')
				else
					@gallery_items = @$element.parents(this.options.gallery_parent_selector).first().find('*[data-toggle="lightbox"][data-gallery="' + @gallery + '"]')
				@gallery_index = @gallery_items.index(@$element)
				$(document).on 'keydown.ekkoLightbox', @navigate.bind(@)

				# add the directional arrows to the modal
				if @options.directional_arrows && @gallery_items.length > 1
					@lightbox_container.append('<div class="ekko-lightbox-nav-overlay"><a href="#" class="'+@strip_stops(@options.left_arrow_class)+'"></a><a href="#" class="'+@strip_stops(@options.right_arrow_class)+'"></a></div>')
					@modal_arrows = @lightbox_container.find('div.ekko-lightbox-nav-overlay').first()
					@lightbox_container.find('a'+@strip_spaces(@options.left_arrow_class)).on 'click', (event) =>
						event.preventDefault()
						do @navigate_left
					@lightbox_container.find('a'+@strip_spaces(@options.right_arrow_class)).on 'click', (event) =>
						event.preventDefault()
						do @navigate_right

			if @options.type
				if @options.type == 'image'
					@preloadImage(@options.remote, true)
				else if @options.type == 'youtube' && video_id = @getYoutubeId(@options.remote)
					@showYoutubeVideo(video_id)
				else if @options.type == 'vimeo'
					@showVimeoVideo(@options.remote)
				else if @options.type == 'instagram'
					@showInstagramVideo(@options.remote);
				else if @options.type == 'url'
					@loadRemoteContent(@options.remote);
				else if @options.type == 'video'
					@showVideoIframe(@options.remote)
				else
					@error "Could not detect remote target type. Force the type using data-type=\"image|youtube|vimeo|instagram|url|video\""

			else
				@detectRemoteType(@options.remote)

	strip_stops: (str) ->
		str.replace(/\./g, '')

	strip_spaces: (str) ->
		str.replace(/\s/g, '')

	isImage: (str) ->
		str.match(/(^data:image\/.*,)|(\.(jp(e|g|eg)|gif|png|bmp|webp|svg)((\?|#).*)?$)/i)

	isSwf: (str) ->
		str.match(/\.(swf)((\?|#).*)?$/i)

	getYoutubeId: (str) ->
		match = str.match(/^.*(youtu.be\/|v\/|u\/\w\/|embed\/|watch\?v=|\&v=)([^#\&\?]*).*/);
		if match && match[2].length == 11 then match[2] else false

	getVimeoId: (str) ->
		if str.indexOf('vimeo') > 0 then str else false

	getInstagramId: (str) ->
		if str.indexOf('instagram') > 0 then str else false

	navigate : ( event ) ->

		event = event || window.event;
		if event.keyCode == 39 || event.keyCode == 37
			if event.keyCode == 39
				do @navigate_right
			else if event.keyCode == 37
				do @navigate_left

	navigateTo: (index) ->

		return @ if index < 0 or index > @gallery_items.length-1

		@showLoading()

		@gallery_index = index

		@$element = $(@gallery_items.get(@gallery_index))
		@updateTitleAndFooter()

		src = @$element.attr('data-remote') || @$element.attr('href')
		@detectRemoteType(src, @$element.attr('data-type') || false)

		if @gallery_index + 1 < @gallery_items.length
			next = $(@gallery_items.get(@gallery_index + 1), false)
			src = next.attr('data-remote') || next.attr('href')
			if next.attr('data-type') == 'image' || @isImage(src)
				@preloadImage(src, false)


	navigate_left: ->

		if @gallery_items.length == 1 then return
		if @gallery_index == 0 then @gallery_index = @gallery_items.length-1 else @gallery_index-- #circular

		@options.onNavigate.call(@, 'left', @gallery_index)
		@navigateTo(@gallery_index)

	navigate_right: ->

		if @gallery_items.length == 1 then return
		if @gallery_index == @gallery_items.length-1 then @gallery_index = 0 else @gallery_index++ #circular

		@options.onNavigate.call(@, 'right', @gallery_index)
		@navigateTo(@gallery_index)

	detectRemoteType: (src, type) ->

		type = type || false

		if type == 'image' || @isImage(src)
			@options.type = 'image'
			@preloadImage(src, true)

		else if type == 'youtube' || video_id = @getYoutubeId(src)
			@options.type = 'youtube'
			@showYoutubeVideo(video_id)

		else if type == 'vimeo' || video_id = @getVimeoId(src)
			@options.type = 'vimeo'
			@showVimeoVideo(video_id)

		else if type == 'instagram' || video_id = @getInstagramId(src)
			@options.type = 'instagram'
			@showInstagramVideo(video_id)

		else if type == 'video'
			@options.type = 'video'
			@showVideoIframe(video_id)

		else
			@options.type = 'url'
			@loadRemoteContent(src)

	updateTitleAndFooter: ->
		header = @modal_content.find('.modal-header')
		footer = @modal_content.find('.modal-footer')
		title = @$element.data('title') || ""
		caption = @$element.data('footer') || ""

		if title or @options.always_show_close then header.css('display', '').find('.modal-title').html(title || "&nbsp;") else header.css('display', 'none')
		if caption then footer.css('display', '').html(caption) else footer.css('display', 'none')
		@

	showLoading : ->
		@lightbox_body.html '<div class="modal-loading">'+@options.loadingMessage+'</div>'
		@

	showYoutubeVideo : (id) ->
		width = @checkDimensions( @$element.data('width') || 560 )
		height = width / ( 560/315 ) # aspect ratio
		@showVideoIframe('//www.youtube.com/embed/' + id + '?badge=0&autoplay=1&html5=1', width, height)

	showVimeoVideo : (id) ->
		width = @checkDimensions( @$element.data('width') || 560 )
		height = width / ( 500/281 ) # aspect ratio
		@showVideoIframe(id + '?autoplay=1', width, height)

	showInstagramVideo : (id) ->
		# instagram load their content into iframe's so this can be put straight into the element
		width = @checkDimensions @$element.data('width') || 612
		@resize width
		height = width + 80
		@lightbox_body.html '<iframe width="'+width+'" height="'+height+'" src="' + @addTrailingSlash(id) + 'embed/" frameborder="0" allowfullscreen></iframe>'
		@options.onContentLoaded.call(@)
		@modal_arrows.css 'display', 'none' if @modal_arrows #hide the arrows when showing video

	showVideoIframe: (url, width, height) -> # should be used for videos only. for remote content use loadRemoteContent (data-type=url)
		height = height || width # default to square
		@resize width
		@lightbox_body.html '<div class="embed-responsive embed-responsive-16by9"><iframe width="' + width + '" height="' + height + '" src="' + url + '" frameborder="0" allowfullscreen class="embed-responsive-item"></iframe></div>'
		@options.onContentLoaded.call(@)
		@modal_arrows.css 'display', 'none' if @modal_arrows #hide the arrows when showing video
		@

	loadRemoteContent : (url) ->
		width = @$element.data('width') || 560
		@resize width

		disableExternalCheck = @$element.data('disableExternalCheck') || false

		# external urls are loading into an iframe
		if !disableExternalCheck && !@isExternal(url)
			@lightbox_body.load url, $.proxy =>
		  		@$element.trigger('loaded.bs.modal')

		else
			@lightbox_body.html '<iframe width="'+width+'" height="'+width+'" src="' + url + '" frameborder="0" allowfullscreen></iframe>'
			@options.onContentLoaded.call(@)

		@modal_arrows.css 'display', 'none' if @modal_arrows #hide the arrows when remote content
		@

	isExternal : (url) ->
		match = url.match(/^([^:\/?#]+:)?(?:\/\/([^\/?#]*))?([^?#]+)?(\?[^#]*)?(#.*)?/);
		return true if typeof match[1] == "string" && match[1].length > 0 && match[1].toLowerCase() != location.protocol
		return true if typeof match[2] == "string" && match[2].length > 0 && match[2].replace(new RegExp(":("+{"http:":80,"https:":443}[location.protocol]+")?$"), "") != location.host
		false

	error : ( message ) ->
		@lightbox_body.html message
		@

	preloadImage : ( src, onLoadShowImage) ->

		img = new Image()
		if !onLoadShowImage? || onLoadShowImage == true
			img.onload = =>
				image = $('<img />')
				image.attr('src', img.src)
				image.addClass('img-responsive')
				@lightbox_body.html image
				@modal_arrows.css 'display', 'block' if @modal_arrows
				image.load =>
					@resize img.width
					@options.onContentLoaded.call(@)
			img.onerror = =>
				@error 'Failed to load image: ' + src

		img.src = src
		img

	resize : ( width ) ->
		#resize the dialog based on the width given, and adjust the directional arrow padding
		width_total = width + @border.left + @padding.left + @padding.right + @border.right
		@modal_dialog.css('width', 'auto') .css('max-width', width_total);

		@lightbox_container.find('a').css 'line-height', ->
			$(@).parent().height() + 'px'
		@

	checkDimensions: (width) ->
		#check that the width given can be displayed, if not return the maximum size that can be

		width_total = width + @border.left + @padding.left + @padding.right + @border.right
		body_width = document.body.clientWidth

		if width_total > body_width
			width = @modal_body.width()

		width

	close : ->
		@modal.modal('hide');

	addTrailingSlash: (url) ->
		if url.substr(-1) != '/'
			url += '/'
		url
}

$.fn.ekkoLightbox = ( options ) ->
	@each ->

		$this = $(this)
		options = $.extend({
			remote : $this.attr('data-remote') || $this.attr('href')
			gallery_parent_selector : $this.attr('data-parent')
			type : $this.attr('data-type')
		}, options, $this.data())
		new EkkoLightbox(@, options)
		@

$.fn.ekkoLightbox.defaults = {
	gallery_parent_selector: 'document.body'
	left_arrow_class: '.glyphicon .glyphicon-chevron-left' #include class . here - they are stripped out later
	right_arrow_class: '.glyphicon .glyphicon-chevron-right' #include class . here - they are stripped out later
	directional_arrows: true #display the left / right arrows or not
	type: null #force the lightbox into image / youtube mode. if null, or not image|youtube|vimeo; detect it
	always_show_close: true #always show the close button, even if there is no title
	loadingMessage: 'Loading...',
	onShow : ->
	onShown : ->
	onHide : ->
	onHidden : ->
	onNavigate : ->
	onContentLoaded : ->
}
