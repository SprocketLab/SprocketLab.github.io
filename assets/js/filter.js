window.onload = function() {
    console.log('Filter script loaded!');
    var buttons = document.querySelectorAll('.filter-btn');
    var publications = document.querySelectorAll('.list__item, .grid__item');
    var yearHeadings = document.querySelectorAll('.year-heading');
    
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
        var visibleByYear = {};
        
        publications.forEach(function(pub) {
          // Get the year from the year container
          var yearContainer = pub.closest('.publications-year');
          var year = yearContainer ? yearContainer.getAttribute('data-year') : null;
          
          if (!visibleByYear[year]) {
            visibleByYear[year] = 0;
          }
          
          if (filter === 'all') {
            pub.style.display = 'block';
            visibleByYear[year]++;
            return;
          }
          
          // Get categories from the data attribute
          var categories = pub.getAttribute('data-categories');
          
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
            visibleByYear[year]++;
          } else {
            pub.style.display = 'none';
          }
        });
        
        // Show/hide year headings based on visible publications
        yearHeadings.forEach(function(heading) {
          var year = heading.textContent.trim();
          var yearContainer = document.querySelector('.publications-year[data-year="' + year + '"]');
          
          if (visibleByYear[year] && visibleByYear[year] > 0) {
            heading.style.display = 'block';
            if (yearContainer) yearContainer.style.display = 'block';
          } else {
            heading.style.display = 'none';
            if (yearContainer) yearContainer.style.display = 'none';
          }
        });
      });
    });
  };