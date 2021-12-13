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
Many new tools for data storage and processing emerged recently, they are optimized for a variety different use cases, and they no
longer neatly fit into traditional categories. For example, *Redis* are datastores that are also used as message queues, *Kafka*
are message queues with database-like durability guarantees.

We will focus on 3 concerns that are important in most software systems:

**Reliability**: The system should continue to work *correctly* (performing the correct function at the desired level of performance)
even in the face of *adversity* (hardware or software faults, and even human error).

**Scalability**: As the system *grows* (in data volume, traffic volume, or complexity), there should be reasonable ways of dealing with
that growth.

**Maintainability**: Over time, many different people will work on the system (engineering and operations, both maintaining current
behavior and adapting the system to new use cases), and they should all be able to work on it *productively*.
