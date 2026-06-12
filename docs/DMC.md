# Data Model Canvas

## 1. Users

## Who will use or benefit from the data/product:

- people wanting to counteract against the confirmation bias
- essay **writers** incorporating multiple opinions and perspectives
- **researchers** and **scientists** while writing papers (unable to know everything)
- **forecasters** and **analysts** for aggregating views and forecasting based on several information sources
- **entrepreneurs** looking for business ideas

## 2. Business problem / Research question
Primary problems to solve or contribute to:

- Semantic Divergence vs. Stylistic/Contextual Similarity: How can one retrieve statements, opinions, or arguments regarding the same topic that share a similar linguistic style and context, but carry a fundamentally different or opposing semantic meaning (stance)?

- Infrastructure & Computational Efficiency: How to achieve complex argument and stance mining in a significantly less computationally expensive way than relying on Large Language Models (LLMs) for brute-force generation or extraction?

- Multi-Criteria Query Optimization & Search Space Maneuverability: How to enhance and improve the quality of information retrieval by combining multiple criteria (semantics, stylometrics, argument structures, and propaganda detection)? How can we leverage this multi-stage approach to dynamically adjust the search space, allowing the system to iterate on initial results and execute refined, multi-dimensional queries?

## 3. Business / Scientific value
Key expected value or contribution: 

### Scientific / Technical
- Mitigating the Echo Chamber Effect in AI Retrieval: Traditional semantic search and vector embeddings are mathematically designed to minimize distance between similar texts. While efficient for finding identical contexts, this inherent nature of vector search inevitably retrieves similar opinions and perspectives, reinforcing the confirmation bias instead of counteracting it. This project introduces a method to separate topical similarity from argumentative stance, breaking the algorithmic echo chamber.

- High-Efficiency Algorithmic Alternative to LLMs: Providing a lighter, faster, and cheaper architectural alternative to LLMs for complex text analysis. By decomposing the task into multi-stage feature extraction (SLMs, binary classifiers, stylometry) and vector matching, we want to prove that high-level analytical retrieval does not require billions of parameters.

- Unified Feature-Semantic Embedding Paradigm: Contributing a new framework to the field of Information Retrieval (IR) that shifts away from traditional text augmentation via discrete metadata tagging. Instead of appending external labels, the system constructs a unified joint vector representation (subject + features) using specialized embedding techniques (e.g., custom bi-encoders). This allows semantic context, argument structures, and framing dimensions to be intrinsically encoded within the same dense vector space, enabling multi-dimensional retrieval and stance discovery directly through native vector space mechanics.


### Business
- Value Proposition & Market Adoption (How we drive usage): The system will be positioned as an API-first, framework-agnostic middleware that cuts operational costs and solves the "cognitive blindness" of modern search engines. We encourage adoption by offering an easily integrable micro-service that provides a immediate, measurable ROI: reducing infrastructure bills (API tokens/compute) while delivering multi-layered text analytics that standard search providers cannot offer.

- Core Architectural Validation: To create and validate a multi-criteria search algorithm (Multi-Criteria Retriever) that effectively combines argument classification, frame classification, and vector matching without using large language models.

- Versatile, Domain-Agnostic Matching Infrastructure: While the baseline implementation focuses on a specific scenario—extracting opposing perspectives—the core architecture is decoupled from any single domain. It is designed as a highly adaptable, multi-purpose Matching Engine capable of operating in three distinct environments:

   -  Standalone Transactional/Recommendation Systems: Can be deployed independently in commercial environments, such as e-commerce or publishing (e.g., cross-matching books in a bookstore based on complementary/opposing thematic frames, divergent narrative styles, or alternative ideological stances).

    - Agentic AI & Autonomous Workflows: Serving as a lightweight routing and decision-making engine for AI agents that need to evaluate text properties without spinning up expensive LLM calls.

    - Generative AI Context Optimization (Optional Layer): Acting as an advanced context-balancing pre-processor for RAG systems, ensuring that generative models receive structured and multi-dimensional inputs.

## 4. Data

Which data sources, formats, volume, quality, access.

### Argument mining datasets

1. https://gitlab.com/tomaye/abstrct – AbstRCT; medical dataset
2. https://github.com/LiyingCheng95/IAM – IAM; series of argument mining tasks: claim extraction, stance classification, evidence extraction (argument/no argument), claim-evidence pair extraction
3. https://huggingface.co/datasets/DFKI-SLT/cdcp – Cornell eRulemaking Corpus (CDCP); legal dataset, uses strange scripts to download data
4. https://huggingface.co/datasets/mfromm/AMSR – Argument Mining in Scientific Reviews (AMSR); argument detection, stance detection and joint detection, which is a combination of argumentation and stance detection; token-level and sentence level
5. https://github.com/gnikesh/gunstance – GunStance; Twitter posts on gun control, manually annotated, stance classification
6. https://huggingface.co/datasets/WIBA/WIBA-Corpus – WIBA Corpus; evidence extraction (argument/no argument); requires HuggingFace access request
7. Synthetic, LLM-generated samples based on real samples.
8. Augmentation to provide more samples – remove non-crucial words, swap order of words etc., see text data augmentation for more.

Search was performed with: https://huggingface.co/search/full-text?q=argument+mining&type=dataset

### Information retrieval (IR) datasets

1. https://github.com/beir-cellar/beir

### Frame retrieval

1. https://huggingface.co/datasets/nyu-mll/multi_nli
2. https://cims.nyu.edu/~sbowman/multinli/

## 5. Model

Planned analytical or ML model, approach, algorithms:

1. Vector search + feature extraction
   1. subject extraction: argument mining, propaganda detection
   2. create subject + features representation: bi-encoder
   3. Texts search using keyword API
   4. Custom system for copmarison of input text embedding with top-k texts returned by API embeddings
2. Clustering
   1. binary argument classification
   2. frame classification
   3. embedding pair (frame class, argument)
   4. embedding distance measuring (outcasts can be treated as more interesting)
3. Final stage
   1. matching arguments with counterarguments (texts)
   2. displaying a list of counterarguments to the user

## 6. Evaluation

Metrics, validation strategy, success criteria

### Metrics

- contrargument precision
- time (faster than LLM)

### Validation strategy

Provided an unseen testing set of examples each step of the process satisfies the following criteria.

### Success criteria

## 7. Team, partners, collaborators, consultants

Roles, responsibilities, external partners:

1. Scientific mentor & consultant: PhD Jan Kocoń

2. Team members:
   - Aleksandra Dowgwiłłowicz-Nowicka

   - Jakub Gonczarek

   - Igor Sosnowicz

## 8. Expected benefits for the provider

Benefits, value proposition for the data/product provider:

- Scientific Benefit: New hybrid search algorithm.
  Metric: Scientific publication in a journal and the repository's source code under an open source license.
- Implementation Benefit: Reduced infrastructure costs.
  Metric: Infrastructure maintenance costs (queries) are at least 60% lower compared to a similar solution based on commercial LLM models (e.g., GPT-4).
- User Benefit: Increased objectivity and time savings.
  Metric: Researchers' literature/opinion gathering time is reduced by 40% (verified by survey testing).
  Qualitative Measure: Accuracy of counterarguments (Precision@K) at a minimum level of 85% in expert assessment.

## 9. Project products

Product components:

- Source extraction and filtering module (Hybrid/Keyword/Semantic Search).
- Argument Classification: Yes/No binary model.
- Frame Classification model.
- Generation module for argument-perspective pairs and a vector database.
- Ranking algorithm for pairing arguments with counterarguments.
- API technical documentation.

## 10. Risks (threats and opportunities)

Risks, mitigations, potential upsides:
Technological risk:

- not enough quality data

Financial risk:

- there might be a need to pay for a model/API

Scientific risk:

- system might present worse results than LLMs in terms of precision and time

Social risk:

- low interest in our system from scientists or writers
