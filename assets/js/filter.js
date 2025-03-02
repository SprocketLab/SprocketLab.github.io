window.onload = function() {
    console.log('Filter script loaded!');
    var buttons = document.querySelectorAll('.filter-btn');
    var publications = document.querySelectorAll('.list__item, .grid__item');
    
    // Debug: Log all publications and their categories
    publications.forEach(function(pub) {
      console.log('Publication element:', pub);
      console.log('Categories attribute:', pub.getAttribute('data-categories'));
      console.log('Title:', pub.querySelector('.archive__item-title')?.textContent);
    });
    
    buttons.forEach(function(button) {
      button.addEventListener('click', function() {
        // Remove active class from all buttons
        buttons.forEach(function(btn) {
          btn.classList.remove('active');
        });
        
        // Add active class to clicked button
        this.classList.add('active');
        
        // Get filter value and normalize it
        var filter = this.getAttribute('data-filter').toLowerCase();
        console.log('Filter selected:', filter);
        
        // Show/hide publications based on filter
        var visibleCount = 0;
        publications.forEach(function(pub) {
          if (filter === 'all') {
            pub.style.display = 'block';
            visibleCount++;
            return;
          }
          
          // Get categories from the data attribute
          var categories = pub.getAttribute('data-categories');
          console.log('Publication categories for', pub.querySelector('.archive__item-title')?.textContent || 'unknown', ':', categories);
          
          // If no categories or empty, hide when filtering
          if (!categories) {
            pub.style.display = 'none';
            return;
          }
          
          // Convert categories to lowercase for comparison
          categories = categories.toLowerCase();
          
          // Check if the filter is included in the categories string
          if (categories.includes(filter)) {
            pub.style.display = 'block';
            visibleCount++;
          } else {
            pub.style.display = 'none';
          }
        });
        
        console.log('Visible publications after filtering:', visibleCount);
      });
    });
  };