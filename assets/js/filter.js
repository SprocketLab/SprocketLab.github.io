// assets/js/filter.js
window.onload = function() {
    console.log('Filter script loaded!');
    var buttons = document.querySelectorAll('.filter-btn');
    
    buttons.forEach(function(button) {
      button.addEventListener('click', function() {
        // Remove active class from all buttons
        buttons.forEach(function(btn) {
          btn.classList.remove('active');
        });
        
        // Add active class to clicked button
        this.classList.add('active');
      });
    });
  };