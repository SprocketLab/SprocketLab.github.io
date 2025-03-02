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
    background-color: #f0f8ff !important;
    color: #5aabca !important; 
    border-color: #5aabca !important;
  }
</style>

<script type="text/javascript">
  // Wait for page to load
  window.onload = function() {
    console.log('Page loaded');
    // Get all filter buttons
    var buttons = document.querySelectorAll('.filter-btn');
    console.log('Found buttons:', buttons.length);
    
    // Add click event to each button
    buttons.forEach(function(button) {
      button.addEventListener('click', function() {
        console.log('Button clicked:', this.textContent);
        
        // Remove active class from all buttons
        buttons.forEach(function(btn) {
          btn.classList.remove('active');
        });
        
        // Add active class to clicked button
        this.classList.add('active');
      });
    });
  };
</script>

{% include base_path %}

{% for post in site.publications reversed %}
  {% include archive-single.html %}
{% endfor %}