---
layout: theme
title: Home
permalink: /
show_date: false
---

{% for post in collections.post reversed %}
<a href="{{ post.url }}" class="collection-item">
    <h2>{{ post.data.title }}</h2>
    <p class="post-date">{{ post.date | postDate }}</p>
    <p class="post-preview">{{ post.content | split: "\n" | first | strip_html | newline_to_br | truncate: 1000, "…" }}</p>
</a>
{% endfor %}
