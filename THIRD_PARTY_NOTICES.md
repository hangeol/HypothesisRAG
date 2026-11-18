# Third-party notices

This repository includes code and benchmark data from the projects below. Their original licenses and attribution requirements continue to apply. These notices do not relicense third-party material.

## MIRAGE and MedRAG

- [MIRAGE](https://github.com/gzxiong/MIRAGE): the imported code and benchmark framework are in `MIRAGE/`. The original [NCBI Public Domain Notice](MIRAGE/LICENSE) is retained.
- [MedRAG](https://github.com/gzxiong/MedRAG): the imported retrieval toolkit is in `MIRAGE/MedRAG/`. Its [NCBI Public Domain Notice](MIRAGE/MedRAG/LICENSE) is retained.
- Citation: Guangzhi Xiong et al., *Benchmarking Retrieval-Augmented Generation for Medicine*, Findings of ACL 2024. [Paper](https://aclanthology.org/2024.findings-acl.372/).

These notices cover the upstream projects' own material. The datasets and other bundled components retain their separate terms.

## Medical question-answering datasets

`MIRAGE/benchmark.json` contains the MIRAGE evaluation collection. `data/medqa_train.json` contains MedQA training questions in the repository's JSON format. The dataset contents are unchanged by this notice update.

| Dataset | Included location | Creators and source | Upstream terms |
| --- | --- | --- | --- |
| MedQA | `data/medqa_train.json`; `medqa` in `MIRAGE/benchmark.json` | Di Jin et al.; [MedQA](https://github.com/jind11/MedQA) | [MIT notice](third_party/licenses/MedQA-MIT.txt) |
| MMLU medical subset | `mmlu` in `MIRAGE/benchmark.json` | Dan Hendrycks et al.; [MMLU](https://github.com/hendrycks/test) | [MIT notice](third_party/licenses/MMLU-MIT.txt) |
| PubMedQA | `pubmedqa` in `MIRAGE/benchmark.json` | Qiao Jin et al.; [PubMedQA](https://github.com/pubmedqa/pubmedqa) | [MIT notice](third_party/licenses/PubMedQA-MIT.txt) |
| MedMCQA | `medmcqa` in `MIRAGE/benchmark.json` | Ankit Pal et al.; [MedMCQA](https://github.com/medmcqa/medmcqa) | [MIT notice from the upstream GitHub repository](third_party/licenses/MedMCQA-MIT.txt) |
| BioASQ | `bioasq` in `MIRAGE/benchmark.json` | BioASQ team and contributors; [official datasets](https://participants-area.bioasq.org/datasets/) | [Creative Commons Attribution 2.5](https://creativecommons.org/licenses/by/2.5/); [legal terms](https://creativecommons.org/licenses/by/2.5/legalcode) |

MIRAGE selects and converts the source datasets into a common multiple-choice QA format, including yes/no tasks. Please credit both MIRAGE and the relevant original dataset when using this collection. The original dataset references are also listed in [MIRAGE's README](MIRAGE/README.md).

The MIT copies above were obtained from these upstream license revisions on 2026-11-19:

- [MedQA](https://github.com/jind11/MedQA/blob/27b02f66aac217933c9648a06f82e9f720377925/LICENSE)
- [MMLU](https://github.com/hendrycks/test/blob/4450500f923c49f1fb1dd3d99108a0bd9717b660/LICENSE)
- [PubMedQA](https://github.com/pubmedqa/pubmedqa/blob/1cbae8e92f72f20c8d3747cbb3bf5bc53554d997/LICENSE)
- [MedMCQA](https://github.com/medmcqa/medmcqa/blob/c59ef14ca1990266c4107c7864b45a20fd93e5e0/LICENSE.md)

These revisions identify the reviewed license notices, not the creation dates of the included data. MedMCQA's GitHub notice is MIT; other distributions may display different terms.

## HotpotQA example inherited from MIRAGE

`MIRAGE/docs/js/camera_ready_nd8_dev_sample100.json` is a 100-question example inherited from [MIRAGE's documentation](https://github.com/gzxiong/MIRAGE/blob/392943af99cd94cafd50a0de2e7fca24bbf65494/docs/js/camera_ready_nd8_dev_sample100.json). Its bytes match that upstream file. It contains HotpotQA question IDs, answers, supporting facts, and Wikipedia paragraphs, organized into separate passage and distractor fields. No changes to this data are made by this notice update.

Credit: Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William W. Cohen, Ruslan Salakhutdinov, and Christopher D. Manning, *HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering*, EMNLP 2018, and the contributing Wikipedia authors.

[HotpotQA](https://hotpotqa.github.io/) distributes its dataset and processed Wikipedia under [Creative Commons Attribution-ShareAlike 4.0](https://creativecommons.org/licenses/by-sa/4.0/) ([legal terms](https://creativecommons.org/licenses/by-sa/4.0/legalcode)). The inherited sample remains under those terms. Retain its attribution and license link, identify further modifications, and distribute adaptations under the applicable ShareAlike terms. This data license does not relicense independent project code.

## Retrieved corpora and local execution logs

Obtain retrieval corpora from their original providers and follow the terms applying to each source. In particular, the MedQA source paper describes the textbook collection as provided under a research-use agreement ([Section 3.2.2](https://www.mdpi.com/2076-3417/11/14/6421)). The MIT notice for the MedQA repository should not be treated as a separate grant of rights to textbook content.

The current tree excludes the two historical MedGemma test logs named `medgemma_medqa_rag_results_1756906142.json` and `medgemma_medqa_rag_results_1756906592.json`, which contained retrieved textbook passages and prompts repeating them. Matching MedGemma result logs are ignored by Git. Keep raw corpus text and full prompts containing it in local output directories; use document IDs and aggregate results for public logs unless the source terms allow redistribution.

This is a change to the current distribution. Earlier Git commits have not been rewritten.

## Other bundled software

The imported documentation assets retain their own license headers, including Bootstrap, jQuery, Font Awesome, and jqBootstrapValidation. Generated training cache files and installed dependencies are separate from the benchmark datasets above; their upstream software licenses continue to apply.
