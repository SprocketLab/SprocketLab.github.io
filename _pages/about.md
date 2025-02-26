---
permalink: /
title: ""
excerpt: "About us"
author_profile: true
redirect_from:
  - /about/
  - /about.html
---

<div style="margin-bottom: 15px;">
  <p>A research group at the University of Wisconsin-Madison focused on machine learning, data science, and AI.</p>
</div>

<h2 style="margin-top: 20px; margin-bottom: 15px; border-bottom: 1px solid #eee; padding-bottom: 5px;">Research Topics</h2>

<div class="research-topics-grid">
  <div class="research-topic-card">
    <div class="research-topic-image">
      <img src="/images/research_illustration/datacentric.png" alt="Data-centric AI" onerror="this.src='/images/research_illustration/datacentric.png'">
    </div>
    <h3><a href="/">Data-centric AI</a></h3>
  </div>
  
  <div class="research-topic-card">
    <div class="research-topic-image">
      <img src="/images/research_illustration/efficient-learning.png" alt="Data- and Compute-Efficient Learning" onerror="this.src='/images/research_illustration/efficient-learning.png'">
    </div>
    <h3><a href="/">Data- and Compute-Efficient Learning</a></h3>
  </div>

  <div class="research-topic-card">
    <div class="research-topic-image">
      <img src="/images/research_illustration/foundation-models.png" alt="Foundation Models" onerror="this.src='/images/research_illustration/foundation-models.png'">
    </div>
    <h3><a href="/">Foundation Models</a></h3>
  </div>
  
  <div class="research-topic-card">
    <div class="research-topic-image">
      <img src="/images/research_illustration/weak-supervision.png" alt="Weak Supervision" onerror="this.src='/images/research_illustration/weak-supervision.png'">
    </div>
    <h3><a href="/">Weak Supervision</a></h3>
  </div>
</div>

<h2 style="margin-top: 20px; margin-bottom: 15px; border-bottom: 1px solid #eee; padding-bottom: 5px;">News</h2>

<div style="margin-bottom: 15px; background-color: #f8f9fa; padding: 15px; border-radius: 5px;">
  <h3 style="color: #4682B4; margin-bottom: 8px; font-size: 1.2em; margin-top: 0;">February 2025</h3>
  <p style="font-weight: 600; margin-bottom: 5px;">New papers accepted to start the year!</p>
  <ul style="padding-left: 20px; margin-top: 5px; margin-bottom: 0;">
    <li style="margin-bottom: 5px;">ICLR 2025: Changho and John explain how <a href="https://arxiv.org/pdf/2412.03881?" target="_blank">weak-to-strong generalization works</a>—and how to do more of it!</li>
    <li style="margin-bottom: 5px;">NAACL 2025: Jane, Dyah, and Changho introduced an ultra-efficient way to personalize language models.</li>
  </ul>
</div>

<div style="margin-bottom: 15px; background-color: #f8f9fa; padding: 15px; border-radius: 5px;">
  <h3 style="color: #4682B4; margin-bottom: 8px; font-size: 1.2em; margin-top: 0;">December 2024</h3>
  <p style="font-weight: 600; margin-bottom: 5px;">Four new papers accepted at NeurIPS 2024!</p>
  <ul style="padding-left: 20px; margin-top: 5px; margin-bottom: 0;">
    <li style="margin-bottom: 5px;">Brian, Cathy, and Vaishnavi show how to get rid of the LLM in LLM-based annotation. How? <a href="https://arxiv.org/pdf/2407.11004" target="_blank">Distill LLMs into programs</a> (spotlight)!</li>
    <li style="margin-bottom: 5px;">Harit gets a huge boost in auto-labeling by <a href="https://arxiv.org/pdf/2404.16188" target="_blank">learning confidence functions</a>.</li>
    <li style="margin-bottom: 5px;">Changho, Jitian, and Sonia show how to <a href="https://arxiv.org/pdf/2404.08461" target="_blank">adjust zero-shot model predictions quickly and easily</a></li>
    <li style="margin-bottom: 0;">Chris and Jack introduce a new benchmark showcasing <a href="https://arxiv.org/pdf/2501.07727" target="_blank">how valuable weak supervision can be</a></li>
  </ul>
</div>

<h2 style="margin-top: 20px; margin-bottom: 15px; border-bottom: 1px solid #eee; padding-bottom: 5px;">How to Join</h2>

<p>Interested in joining our lab as a UW-Madison undergraduate? Please complete our <a href="https://forms.gle/8dxCSvtiBYdB3EGDA" style="font-weight: bold; color: #4682B4;">application form</a>. We'll contact promising candidates directly when opportunities align with your background and interests.</p>

<style>
.research-topics-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 15px;
  margin: 10px 0;
}

.research-topic-card {
  border: 1px solid #ddd;
  border-radius: 5px;
  overflow: hidden;
  transition: transform 0.3s, box-shadow 0.3s;
}

.research-topic-card:hover {
  transform: translateY(-3px);
  box-shadow: 0 5px 10px rgba(0,0,0,0.1);
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
  padding: 10px;
  margin: 0;
  text-align: center;
}

.research-topic-card h3 a {
  color: #4682B4;
  text-decoration: none;
}

.research-topic-card h3 a:hover {
  text-decoration: underline;
}

@media (max-width: 900px) {
  .research-topics-grid {
    grid-template-columns: repeat(2, 1fr);
  }
}

@media (max-width: 600px) {
  .research-topics-grid {
    grid-template-columns: 1fr;
  }
}
</style>