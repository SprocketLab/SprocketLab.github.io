---
layout: archive
title: "Publications"
permalink: /publications/
author_profile: true
---

<div class="filter-buttons">
  <button class="filter-btn active" data-filter="all">Show all</button>
  <button class="filter-btn" data-filter="nlp">NLP</button>
  <button class="filter-btn" data-filter="llm">LLM</button>
  <button class="filter-btn" data-filter="alignment">Alignment</button>
  <button class="filter-btn" data-filter="robotics">Robotics</button>
</div>

<style>
  .filter-buttons {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin: 20px 0 30px 0;
  }

  .filter-btn {
    padding: 10px 15px;
    background-color: white;
    border: 1px solid #ddd;
    border-radius: 3px;
    cursor: pointer;
    font-size: 16px;
    transition: all 0.3s ease;
    min-width: 80px;
    text-align: center;
  }

  .filter-btn:hover {
    background-color: #f5f5f5;
  }

  .filter-btn.active {
    background-color: #f0f8ff; /* Light background */
    color: #79b8cc; /* Matching the blue-teal color of your publication titles */
    border-color: #79b8cc;
  }
  /* Make "Show all" button match screenshot */
  .filter-btn[data-filter="all"] {
    background-color: #f0f8ff;
    color: #79b8cc;
    border-color: #79b8cc;
  }
</style>

<script>
  document.addEventListener('DOMContentLoaded', function() {
    const filterButtons = document.querySelectorAll('.filter-btn');
    
    // Wrap each publication in a div with data-categories
    const publications = document.querySelectorAll('.archive__item');
    publications.forEach(function(pub) {
      // Get the title element
      const titleElement = pub.querySelector('.archive__item-title');
      if (titleElement) {
        // Wrap the entire publication in a div
        const wrapper = document.createElement('div');
        wrapper.className = 'publication';
        // Assign categories based on title content (you'll need to customize this)
        const title = titleElement.textContent.toLowerCase();
        let categories = [];
        
        if (title.includes('llm') || title.includes('language model')) {
          categories.push('llm');
        }
        if (title.includes('nlp') || title.includes('natural language')) {
          categories.push('nlp');
        }
        if (title.includes('align') || title.includes('preference')) {
          categories.push('alignment');
        }
        if (title.includes('robot')) {
          categories.push('robotics');
        }
        
        wrapper.setAttribute('data-categories', categories.join(','));
        
        // Move the publication into the wrapper
        pub.parentNode.insertBefore(wrapper, pub);
        wrapper.appendChild(pub);
      }
    });
    
    const publications2 = document.querySelectorAll('.publication');
    
    // Add event listeners to filter buttons
    filterButtons.forEach(button => {
      button.addEventListener('click', function() {
        // Remove active class from all buttons
        filterButtons.forEach(btn => btn.classList.remove('active'));
        
        // Add active class to clicked button
        this.classList.add('active');
        
        const filter = this.getAttribute('data-filter');
        
        // Filter publications
        publications2.forEach(pub => {
          const categories = pub.getAttribute('data-categories').split(',');
          
          if (filter === 'all' || categories.includes(filter)) {
            pub.style.display = 'block';
          } else {
            pub.style.display = 'none';
          }
        });
      });
    });
  });
</script>

{% include base_path %}

{% for post in site.publications reversed %}
  {% include archive-single.html %}
{% endfor %}