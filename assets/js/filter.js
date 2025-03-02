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
          if (filter === 'all') {
            pub.style.display = 'block';
            return;
          }
          
          var categories = pub.getAttribute('data-categories');
          console.log('Publication categories:', categories);
          
          // Handle publications with no categories
          if (!categories) {
            pub.style.display = 'none';
            return;
          }
          
          // Split the categories string into an array and trim each value
          categories = categories.split(',').map(function(cat) {
            return cat.trim().toLowerCase();
          });
          
          // Check if the filter is in the categories array
          if (categories.includes(filter.toLowerCase())) {
            pub.style.display = 'block';
          } else {
            pub.style.display = 'none';
          }
        });
      });
    });
  };