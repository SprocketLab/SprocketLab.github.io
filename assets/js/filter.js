window.onload = function() {
    console.log('Filter script loaded!');
    var buttons = document.querySelectorAll('.filter-btn');
    var publications = document.querySelectorAll('.archive__item');
    
    buttons.forEach(function(button) {
      button.addEventListener('click', function() {
        // Remove active class from all buttons
        buttons.forEach(function(btn) {
          btn.classList.remove('active');
        });
        
        // Add active class to clicked button
        this.classList.add('active');
        
        // Get filter value
        var filter = this.getAttribute('data-filter');
        console.log('Filter selected:', filter);
        
        // Show/hide publications based on filter
        publications.forEach(function(pub) {
          var categories = pub.getAttribute('data-categories');
          
          // Handle publications with no categories
          if (!categories) {
            pub.style.display = (filter === 'all') ? 'block' : 'none';
            return;
          }
          
          categories = categories.split(',');
          
          if (filter === 'all' || categories.includes(filter)) {
            pub.style.display = 'block';
          } else {
            pub.style.display = 'none';
          }
        });
      });
    });
  };