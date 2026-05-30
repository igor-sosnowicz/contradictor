# Data Model Canvas

## 1. Users
Who will use or benefit from the data/product:

- people wanting to counteract against the confirmation bias
- essay **writers** incorporating multiple opinions and perspectives
- **researchers** and **scientists** while writing papers (unable to know everything)
- **forecasters** and **analysts** for aggregating views and forecasting based on several information sources
- **entrepreneurs** looking for business ideas

## 2. Business problem / Research question
Primary problem to solve or work on research question to answer:

- How can one retrieve statements/opinions/arguments regarding the same topic, in a similar style but of different meaning (semantically different)?

## 3. Business / Scientific value
Key expected value or contribution:

- 

## 4. Data

Which data sources, formats, volume, quality, access.

### Argument mining datasets

1. https://gitlab.com/tomaye/abstrct – AbstRCT; medical dataset
2. https://github.com/LiyingCheng95/IAM –  IAM; series of argument mining tasks: claim extraction, stance classification, evidence extraction (argument/no argument), claim-evidence pair extraction
3. https://huggingface.co/datasets/DFKI-SLT/cdcp – Cornell eRulemaking Corpus (CDCP); legal dataset, uses strange scripts to download data
4. https://huggingface.co/datasets/mfromm/AMSR – Argument Mining in Scientific Reviews (AMSR); argument detection, stance detection and joint detection, which is a combination of argumentation and stance detection; token-level and sentence level
5. https://github.com/gnikesh/gunstance – GunStance; Twitter posts on gun control, manually annotated, stance classification
6. https://huggingface.co/datasets/WIBA/WIBA-Corpus – WIBA Corpus; evidence extraction (argument/no argument); requires HuggingFace access request
7. Synthetic, LLM-generated samples based on real samples.
8. Augmentation to provide more samples – remove non-crucial words, swap order of words etc., see text data augmentation for more.

Search was performed with: https://huggingface.co/search/full-text?q=argument+mining&type=dataset

### Information retrieval (IR) datasets

1. https://github.com/beir-cellar/beir

## 5. Model

Planned analytical or ML model, approach, algorithms:

1. Argument mining.
   1. claim extraction – 
   2. non-binary stance classification – 
   3. argument extraction – extract spans of text being arguments via argument/no argument classification 
   4. argument-claim pairs matching – 
   5. retrieve claims/arguments regarding the same topic, in a similar style but of different meaning (semantically different)
2. Argument representation – vectorisation, metadata encoding, graph-based argument map creation.
3. Information retrieval and re-ranking – hybrid search: semantic similarity search, keyword search, BM25-based search.

## 6. Evaluation
Metrics, validation strategy, success criteria

### Metrics

- 

### Validation strategy

Provided an unseen testing set of examples each step of the process satisfies the following criteria.

### Success criteria



## 7. Team, partners, collaborators, consultants

Roles, responsibilities, external partners:

1. Scientific mentor & consultant: Dr Eng. Jan Kocoń

2. Team members:

   - Aleksandra Dowgwiłłowicz-Nowicka

   - Jakub Gonczarek

   - Igor Sosnowicz

## 8. Expected benefits for the provider
Benefits, value proposition for the data/product provider

## 9. Project products
Deliverables, outputs, artifacts

## 10. Risks (threats and opportunities)

Risks, mitigations, potential upsides:

### Risks

- Low quality data.

### Opportunities

- Use LLMs to 