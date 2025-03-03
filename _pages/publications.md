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
  
  /* Style for year headings */
  /* Style for year headings */
  .year-heading {
    font-size: 2.5em;
    color: #333;
    margin-top: 2em;
    margin-bottom: 0.8em;
    font-weight: 600;
    border-bottom: 1px solid #eaeaea;
    padding-bottom: 8px;
    width: 100%;
  }
</style>

<script src="{{ site.baseurl }}/assets/js/filter.js"></script>

{% include base_path %}

{% assign grouped_publications = site.publications | group_by_exp: "pub", "pub.date | date: '%Y'" | sort: "name" | reverse %}

{% for year in grouped_publications %}
  <h2 class="year-heading">{{ year.name }}</h2>
  <div class="publications-year" data-year="{{ year.name }}">
    {% for post in year.items %}
      {% include archive-single.html %}
    {% endfor %}
  </div>
{% endfor %}