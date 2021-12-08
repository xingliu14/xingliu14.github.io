---
title: "DDIA Chapter 1: Reliable, Scalable, and Maintainable Applications"
published: true
---

Many applications are data-intensive instead of compute-intensive, for example:
- Store data so that they, or another application, can find it again later (*database*)
- Remember the result of an expensive operation, to speed up reads (*caches*)
- Allow users to search data or filter in carious ways (*search indexes*)
- Send a message to another process, to be handled asynchronously (*stream processing*)
- Periodically crunch a large amount of data (*batch processing*)

When building an application, we need to figure out which tools and which approaches are the most appropriate for the task at hand.

# Thinking About Data Systems
