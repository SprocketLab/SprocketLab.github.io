---
permalink: /
title: ""
excerpt: "About us"
author_profile: true
redirect_from:
  - /about/
  - /about.html
---

<div class="intro-section">
  <h1>Welcome to Sprocket Lab!</h1>
  <p>A research group at the University of Wisconsin-Madison broadly focused on machine learning, LLMs, and data-centric AI.</p>
</div>

<section class="content-section">
  <h2 class="section-heading">Research Topics</h2>

  <div class="research-topics-grid">
    <a href="/" class="research-topic-card">
      <div class="research-topic-image">
        <img src="/images/research_illustration/datacentric.png" alt="Data-centric AI" onerror="this.src='/images/research_illustration/datacentric.png'">
      </div>
      <h3>Data-centric AI</h3>
    </a>
    
    <a href="/" class="research-topic-card">
      <div class="research-topic-image">
        <img src="/images/research_illustration/efficient-learning.png" alt="Data- and Compute-Efficient Learning" onerror="this.src='/images/research_illustration/efficient-learning.png'">
      </div>
      <h3>Data- and Compute-Efficient Learning</h3>
    </a>

    <a href="/" class="research-topic-card">
      <div class="research-topic-image">
        <img src="/images/research_illustration/foundation-models.png" alt="Foundation Models" onerror="this.src='/images/research_illustration/foundation-models.png'">
      </div>
      <h3>Foundation Models</h3>
    </a>
    
    <a href="/" class="research-topic-card">
      <div class="research-topic-image">
        <img src="/images/research_illustration/weak-supervision.png" alt="Weak Supervision" onerror="this.src='/images/research_illustration/weak-supervision.png'">
      </div>
      <h3>Weak Supervision</h3>
    </a>
  </div>
</section>

<section class="content-section alt-background">
  <h2 class="section-heading">News</h2>
  
  <div class="news-item">
    <h3 class="news-date">February 2025</h3>
    <p class="news-content">New papers accepted to start the year!</p>
    <ul>
      <li>ICLR 2025: Changho and John explain how <a href="https://arxiv.org/pdf/2412.03881?" target="_blank">weak-to-strong generalization works</a>—and how to do more of it!</li>
      <li>NAACL 2025: Jane, Dyah, and Changho introduced an ultra-efficient way to personalize language models.</li>
    </ul>
  </div>
  
  <div class="news-item">
    <h3 class="news-date">December 2024</h3>
    <p class="news-content">Four new papers accepted at NeurIPS 2024!</p>
    <ul>
      <li>Brian, Cathy, and Vaishnavi show how to get rid of the LLM in LLM-based annotation. How? <a href="https://arxiv.org/pdf/2407.11004" target="_blank">Distill LLMs into programs</a> (spotlight)!</li>
      <li>Harit gets a huge boost in auto-labeling by <a href="https://arxiv.org/pdf/2404.16188" target="_blank">learning confidence functions</a>.</li>
      <li>Changho, Jitian, and Sonia show how to <a href="https://arxiv.org/pdf/2404.08461" target="_blank">adjust zero-shot model predictions quickly and easily</a></li>
      <li>Chris and Jack introduce a new benchmark showcasing <a href="https://arxiv.org/pdf/2501.07727" target="_blank">how valuable weak supervision can be</a></li>
    </ul>
  </div>
</section>

<section class="content-section">
  <h2 class="section-heading">How to Join</h2>
  <p>Interested in joining our lab as a UW-Madison undergraduate? Please complete our application form. We'll contact promising candidates directly when opportunities align with your background and interests.</p>
  <p>Note: We strongly recommend using the form rather than email, as email inquiries may go unnoticed due to high volume.</p>
  
  <a href="https://forms.gle/8dxCSvtiBYdB3EGDA" class="cta-button" target="_blank">Apply to Join Our Lab</a>
</section>

<style>
/* General section styling */
.intro-section {
  margin-bottom: 40px;
}

.intro-section h1 {
  margin-bottom: 15px;
  font-size: 2em;
}

.intro-section p {
  font-size: 1.1em;
  line-height: 1.5;
}

.content-section {
  margin: 60px 0;
  padding: 30px;
  border-radius: 8px;
}

.alt-background {
  background-color: #f8f9fa;
}

.section-heading {
  margin-bottom: 30px;
  font-size: 1.8em;
  border-bottom: 2px solid #eaeaea;
  padding-bottom: 10px;
}

/* Research Topics Grid */
.research-topics-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr); /* Updated to 2 columns for better sizing */
  gap: 25px;
  margin: 30px 0;
}

.research-topic-card {
  border: 1px solid #ddd;
  border-radius: 8px;
  overflow: hidden;
  transition: transform 0.3s, box-shadow 0.3s;
  text-decoration: none;
  color: inherit;
  display: block;
}

.research-topic-card:hover {
  transform: translateY(-5px);
  box-shadow: 0 8px 20px rgba(0,0,0,0.1);
}

.research-topic-image {
  height: 220px;
  overflow: hidden;
}

.research-topic-image img {
  width: 100%;
  height: 100%;
  object-fit: contain;
  object-position: center;
}

.research-topic-card h3 {
  padding: 20px;
  margin: 0;
  text-align: center;
  color: #4682B4;
  font-size: 1.2em;
}

/* News Items */
.news-item {
  margin-bottom: 40px;
}

.news-date {
  font-size: 1.4em;
  color: #4682B4;
  margin-bottom: 10px;
  font-weight: 600;
}

.news-content {
  font-weight: 600;
  margin-bottom: 10px;
}

.news-item ul {
  padding-left: 20px;
}

.news-item ul li {
  margin-bottom: 8px;
  line-height: 1.5;
}

/* CTA Button */
.cta-button {
  display: inline-block;
  background-color: #4682B4;
  color: white;
  padding: 12px 25px;
  border-radius: 5px;
  text-decoration: none;
  font-weight: 600;
  margin-top: 20px;
  transition: background-color 0.3s;
}

.cta-button:hover {
  background-color: #3a6d99;
  text-decoration: none;
}

/* Make sure it's responsive on smaller screens */
@media (max-width: 900px) {
  .research-topics-grid {
    grid-template-columns: 1fr; /* 1 column on medium and small screens */
  }
  
  .content-section {
    padding: 20px;
  }
}

@media (max-width: 600px) {
  .section-heading {
    font-size: 1.5em;
  }
  
  .news-date {
    font-size: 1.2em;
  }
}
</style>