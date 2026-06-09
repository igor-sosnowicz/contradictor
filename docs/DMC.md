# Data Model Canvas

## 1. Users

## Who will use or benefit from the data/product:

- people wanting to counteract against the confirmation bias
- essay **writers** incorporating multiple opinions and perspectives
- **researchers** and **scientists** while writing papers (unable to know everything)
- **forecasters** and **analysts** for aggregating views and forecasting based on several information sources
- **entrepreneurs** looking for business ideas

## 2. Business problem / Research question

Primary problem to solve or work on research question to answer:

- How can one retrieve statements/opinions/arguments regarding the same topic, in a similar style but of different meaning (semantically different)?
- How to achieve the goal in a less computationally expensive way than LLMs do?

## 3. Business / Scientific value

Key expected value or contribution:

### Scientific

- multi-criteria retriever

### Business

- To create and validate a multi-criteria search algorithm (Multi-Criteria Retriever) that effectively combines argument classification, frame classification, and vector matching without using large language models.

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
