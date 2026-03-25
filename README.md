<p align="center">
Poisoning-Resilient Isolated Stratified Memory for LLM Agents
</p>

<p align="justify">
Agentic artificial intelligence systems built upon large language models (LLMs) are increasingly equipped with long-term memory to retain user preferences and maintain contextual continuity across sessions. Whilst this capability enhances agent utility, it introduces a significant attack surface: adversaries may exploit indirect prompt injection or poisoned external data to embed malicious content within an agent's memory, enabling harmful behaviour to resurface long after the originating interaction has concluded. Existing defences largely neglect how untrusted information is stored, classified, and retrieved over time, leaving agentic systems vulnerable to cross-session injection persistence.
</p>

<p align="justify">
This project proposes a security-aware memory hierarchy that reconceptualises agent memory as a security boundary rather than a passive datastore. The architecture partitions memory into three tiers — a scratch tier for transient, zero-persistence data; a session tier governed by strict time-to-live policies; and a long-term tier protected through cryptographic signing and integrity verification. An automated classifier routes each incoming memory item to the appropriate tier based on source metadata and semantic analysis, preventing untrusted content from migrating into long-term storage. Tier isolation and mandatory signature verification together ensure that poisoned content cannot persist across sessions or be elevated in privilege.
</p>

<p align="justify">
The system is evaluated against established memory-poisoning attack patterns drawn from recent red-teaming literature. Preliminary analysis indicates that the proposed architecture can reduce successful cross-session poisoning to near zero, with modest overhead relative to conventional flat memory designs. This work offers a practical, architecture-level complement to existing prompt-level filters and model alignment techniques, contributing towards the secure deployment of agentic AI systems in real-world environments.
</p>
