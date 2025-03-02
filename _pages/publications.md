---
layout: archive
title: "Publications"
permalink: /publications/
author_profile: true
---

<div class="filter-buttons">
  <button class="filter-btn active" data-filter="all">Show all</button>
  
  {% assign all_categories = "" | split: "" %}
  {% for post in site.publications %}
    {% if post.categories %}
      {% assign all_categories = all_categories | concat: post.categories %}
    {% endif %}
  {% endfor %}
  
  {% assign unique_categories = all_categories | uniq | sort %}
  {% for category in unique_categories %}
    <button class="filter-btn" data-filter="{{ category }}">{{ category | capitalize }}</button>
  {% endfor %}
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

<script src="{{ site.baseurl }}/assets/js/filter.js"></script>

{% include base_path %}

{% for post in site.publications reversed %}
  {% include archive-single.html %}
{% endfor %}