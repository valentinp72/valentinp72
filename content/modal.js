// Script to enable modal dialog boxes.
// @valentinp72

// minimal HTML content:
// <a id="open-modal1">Open box</a>
// <div id="modal1" class="modal">
// 	<div class="close-button">Close box</div>
// 	<p>Hello world!</p>
// </div>

var modal_boxes = [];
for (const modal of document.getElementsByClassName('modal')) {
	modal_boxes.push({
		'open-btn': document.getElementById('open-' + modal.id),
		'close-btn': modal.getElementsByClassName('close-button')[0],
		'modal': modal
	});
}

const body = document.querySelector("body");
var scrollTop = 0;

open = function(modal) {
	// showing the modal box and disabling scroll without removing the scrollbar
	scrollTop = window.scrollY;
	document.body.style.top = '-' + scrollTop + 'px';
	modal.style.display = "flex";
	document.body.style.position = "fixed";
	document.body.style.width = "100%";
	document.body.style.overflowY = "scroll";
}

close = function(modal) {
	// closing the modal box et re-enabling scrolling
	modal.style.display = "none";
	document.body.style.top = '0px';
	document.body.style.position = "static";
	document.body.style.width = "100%";
	document.body.style.overflowY = "scroll";
	window.scroll(0, scrollTop);
}

// click events
for (box of modal_boxes) {
	const modal = box['modal'];
	box['open-btn'].onclick = function() {
		open(modal);
	}
	box['close-btn'].onclick = function() {
		close(modal);
	}
}
window.onclick = function(event) {
	for (box of modal_boxes) {
		if (event.target == box['modal']) {
			close(box['modal']);
		}
	}
} 
