---
permalink: /
title: "Sprocket Lab"
excerpt: "About us"
author_profile: true
redirect_from:
  - /about/
  - /about.html
---

Welcome to Sprocket Lab!

# Research Topics

<div class="research-topics-grid">
  <div class="research-topic-card">
    <div class="research-topic-image">
      <img src="/images/research-causality.jpg" alt="Data-centric AI" onerror="this.src='/images/sprocket-logo.png'">
    </div>
    <h3><a href="/research/causality">Data-centric AI</a></h3>
  </div>
  
  <div class="research-topic-card">
    <div class="research-topic-image">
      <img src="/images/research-fairness.jpg" alt="Data- and Compute-Efficient Learning and Adaptation" onerror="this.src='/images/profile.png'">
    </div>
    <h3><a href="/research/fairness">Data- and Compute-Efficient Learning and Adaptation</a></h3>
  </div>
  
  <div class="research-topic-card">
    <div class="research-topic-image">
      <img src="/images/research-healthcare.jpg" alt="Weak Supervision" onerror="this.src='/images/profile.png'">
    </div>
    <h3><a href="/research/healthcare">Weak Supervision</a></h3>
  </div>
</div>

# News
- **February 2025**: New papers accepted to start the year!
  - ICLR 2025: Changho and John explain how [weak-to-strong generalization works](https://arxiv.org/pdf/2412.03881?)---and how to do more of it!
  - NAACL 2025: Jane, Dyah, and Changho introduced an ultra-efficient way to personalize language models.
- **December 2024**: Four new papers accepted at NeurIPS 2024!
  - Brian, Cathy, and Vaishnavi show how to get rid of the LLM in LLM-based annotation. How? [Distill LLMs into programs](https://arxiv.org/pdf/2407.11004) (spotlight)!
  - Harit gets a huge boost in auto-labeling by [learning confidence functions](https://arxiv.org/pdf/2404.16188).
  - Changho, Jitian, and Sonia show how to [adjust zero-shot model predictions quickly and easily](https://arxiv.org/pdf/2404.08461)
  - Chris and Jack introduce a new benchmark showcasing [how valueable weak supervision can be](https://arxiv.org/pdf/2501.07727)

# How to Join
Interested in joining our lab as a UW-Madison undergraduate? Please complete our [application form](https://forms.gle/8dxCSvtiBYdB3EGDA). We'll contact promising candidates directly when opportunities align with your background and interests. Note: We strongly recommend using the form rather than email, as email inquiries may go unnoticed due to high volume.

<style>
.research-topics-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr); /* Exactly 3 columns */
  gap: 20px;
  margin: 30px 0;
}

.research-topic-card {
  border: 1px solid #ddd;
  border-radius: 5px;
  overflow: hidden;
  transition: transform 0.3s, box-shadow 0.3s;
}

.research-topic-card:hover {
  transform: translateY(-5px);
  box-shadow: 0 5px 15px rgba(0,0,0,0.1);
}

.research-topic-image {
  height: 200px;
  overflow: hidden;
}

.research-topic-image img {
  width: 100%;
  height: 100%;
  object-fit: cover;
  object-position: center;
}

.research-topic-card h3 {
  padding: 15px;
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

/* Make sure it's responsive on smaller screens */
@media (max-width: 900px) {
  .research-topics-grid {
    grid-template-columns: repeat(2, 1fr); /* 2 columns on medium screens */
  }
}

@media (max-width: 600px) {
  .research-topics-grid {
    grid-template-columns: 1fr; /* 1 column on small screens */
  }
}
</style>