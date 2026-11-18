# Third-party notices

HYPER includes the code components identified below. Evaluation questions, retrieval corpus text, model weights, and experiment logs are not included in the current source tree. Downloaded resources retain their original terms; these notices do not relicense them.

## Included code

- [MIRAGE](https://github.com/gzxiong/MIRAGE) and [MedRAG](https://github.com/gzxiong/MedRAG), by Guangzhi Xiong and collaborators: the retrieval utilities are under `MIRAGE/MedRAG/`, and the MedRAG answer prompt is retained in `hyper/prompts.py`. Their original [MIRAGE notice](MIRAGE/LICENSE) and [MedRAG notice](MIRAGE/MedRAG/LICENSE) are preserved.
- Citation: Guangzhi Xiong et al., *Benchmarking Retrieval-Augmented Generation for Medicine*, Findings of ACL 2024. [Paper](https://aclanthology.org/2024.findings-acl.372/).

The upstream NCBI Public Domain Notice applies to the upstream authors' material, not automatically to every dataset or corpus used by the code.

## Evaluation questions downloaded separately

`scripts/prepare_data.py` downloads the unchanged MIRAGE evaluation collection from revision `392943af99cd94cafd50a0de2e7fca24bbf65494` and checks SHA-256 `6f7f08c64cd2efe02a5d0c247229813c90db345d9dd6e3a451b5d24146d0f8fa`. It writes an ignored local file, `MIRAGE/benchmark.json`.

| Dataset | Creators and source | Upstream terms |
| --- | --- | --- |
| MedQA | Di Jin et al.; [MedQA](https://github.com/jind11/MedQA) | [MIT notice](third_party/licenses/MedQA-MIT.txt) |
| MMLU medical subset | Dan Hendrycks et al.; [MMLU](https://github.com/hendrycks/test) | [MIT notice](third_party/licenses/MMLU-MIT.txt) |
| PubMedQA | Qiao Jin et al.; [PubMedQA](https://github.com/pubmedqa/pubmedqa) | [MIT notice](third_party/licenses/PubMedQA-MIT.txt) |
| MedMCQA | Ankit Pal et al.; [MedMCQA](https://github.com/medmcqa/medmcqa) | [MIT notice from the upstream GitHub repository](third_party/licenses/MedMCQA-MIT.txt) |
| BioASQ | BioASQ team and contributors; [official datasets](http://participants-area.bioasq.org/datasets/) | [CC BY 2.5](https://creativecommons.org/licenses/by/2.5/) |

License copies are retained for attribution and reference even though dataset contents are downloaded separately.

## Retrieval resources and models

The default corpus is [MedRAG/Textbooks](https://huggingface.co/datasets/MedRAG/textbooks), associated with the MedQA textbook collection. Its source and access conditions must be checked separately; the code's public-domain notice does not grant permission to redistribute textbook content. The corpus and generated outputs remain local.

The retrieval encoder is [MedCPT](https://huggingface.co/ncbi/MedCPT-Query-Encoder). Llama and Qwen model weights are obtained from their respective model repositories and remain subject to their own licenses and access conditions.

## Paper illustration

`assets/hyper-overview.png` reproduces the pipeline panel of Figure 1(a) from the HYPER paper by Hangeol Chang, Changsun Lee, Seungjoon Rho, Junho Yeo, and Jong Chul Ye.
