# Decision Notes (My Implementation Choices)

This file explains the main implementation decisions I took in this project.

## 1 Kept the ingestion pipeline local and deterministic

I decided to run embeddings locally (`all-MiniLM-L6-v2`) and store vectors in ChromaDB.  
Reason: easier setup, no embedding API cost, and stable behavior for evaluation.

## 2 Changed chunking for Markdown to be section-aware

At first, fixed-size chunking created noisy outputs for policy-style docs.  
I updated Markdown chunking to split by `##` sections first, then only use fixed-size overlap if a section is too long.  
Reason: this keeps topics (Return Policy, Loyalty Program, etc.) coherent and improves retrieval precision.

## 3 I route first, then retrieve only when needed

The assignment pipeline says: ingest -> memory -> decide route -> retrieve/tool -> generate final response.  
I followed that, and intentionally avoided embedding every message.

- Name recall/greeting does not need vector search.
- Tool queries (order/product) do not need vector search.
- Only document/knowledge questions run retrieval.

Reason: lower latency, lower compute cost, and fewer irrelevant sources.

## 4 Kept tool data independent from RAG context

Order and product tools are based only on `orders.json` and `products.json`.  
I explicitly separated tool output from uploaded document context.

Reason: avoid accidental blending (for example, product responses inheriting company details from RAG history).

## 5 Made product routing dynamic from the catalog file

Instead of hardcoding a fixed keyword list, I derive searchable catalog terms from `products.json` at runtime.

Reason: if a new product is added, router behavior updates automatically without code changes.

## 6 I supported practical follow-ups in both RAG and tools

- For RAG follow-ups, I expand short questions with recent user context before retrieval.
- For product follow-ups like "show cheaper options", I reuse prior catalog context.

Reason: this improves natural conversational continuity while keeping routing explicit.

## 7 I did not force every final message through the LLM

The pipeline statement can be interpreted as "final response generated using LLM".  
In practice, I used deterministic responses for some routes (greeting/name recall/tool formatting) and used LLM where it adds value (knowledge/direct generation).

Reason: cost control, reliability, and predictable outputs for tool/memory-specific queries.

## 8 Added robust fallback behavior

If the model provider is unavailable, I avoid raw chunk dumps and use structured extractive fallback for knowledge answers.

Reason: user still gets a useful and grounded response even during provider outages or rate spikes.

