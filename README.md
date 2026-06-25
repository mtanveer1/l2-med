# I²-Med: Interpretable Medical Inference through Visual-Guided Dynamic Logits Calibration

<div align="center">

**MICCAI 2026 (Accepted)**

M. Sajid, Shreeyut Maheshwari, Akshat Mishra, Ritik Mishra, M. Tanveer

[Paper (Coming Soon)]() | [ArXiv (Coming Soon)]()

</div>

---

## 📢 News

- **[2026.06]** Source code and pretrained models are publicly released.
- **[2026.06]** 🎉 **I²-Med** has been accepted at **MICCAI 2026**.

---

## Overview

Medical Vision-Language Models (VLMs) have demonstrated remarkable capabilities in medical image understanding and reasoning. However, their predictions often lack visual faithfulness, leading to unreliable explanations and hallucinated reasoning.

**I²-Med** is a **training-free, inference-time framework** that improves the interpretability of medical VLMs through **Visual-Guided Dynamic Logits Calibration**. Rather than fine-tuning the underlying VLM, I²-Med dynamically adjusts token logits during decoding based on visual grounding, resulting in more faithful reasoning and reduced hallucinations while preserving the original model parameters.

<p align="center">
<img src="assets/Architecture.png" width="900">
</p>
<p align="center">
  <b>Fig.:</b> Overview of Proposed I<sup>2</sup>-Med Model.
</p>

---

## Highlights

- ✅ **Training-free** framework
- ✅ Works entirely at **inference time**
- ✅ No fine-tuning of the underlying VLM
- ✅ Reduces visual hallucinations
- ✅ Improves visual faithfulness and reasoning consistency
- ✅ Applicable to existing medical Vision-Language Models

---

## Citation

If you find this repository useful in your research, please consider citing our paper.

### BibTeX

```bibtex
@inproceedings{sajid2026i2med,
  title={I$^2$-Med: Interpretable Medical Inference through Visual-Guided Dynamic Logits Calibration},
  author={M. Sajid and Shreeyut Maheshwari and Akshat Mishra and Ritik Mishra and M. Tanveer},
  booktitle={Medical Image Computing and Computer Assisted Intervention (MICCAI)},
  year={2026},
  publisher={Springer}
}
```

### Plain Text Reference

```
M. Sajid, Shreeyut Maheshwari, Akshat Mishra, Ritik Mishra, and M. Tanveer.
I²-Med: Interpretable Medical Inference through Visual-Guided Dynamic Logits Calibration.
In Proceedings of the 29th International Conference on Medical Image Computing and Computer Assisted Intervention (MICCAI), Springer, 2026.
```

---

## To Run 
1. Download the model weights of the [VLM](https://huggingface.co/yuxianglai117/Med-R1/tree/main) and [BioMedCLIP](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224)
2. Set the parameters in run.sh
3. Use the following command
```bash
bash run.sh
```
---

## Interpretability
Here you can see the qualitative results from our Logits calliberation method . Our method improve the Reasoning in comparision to the base model
1. <br> <img width="900" height="506" alt="image" src="https://github.com/user-attachments/assets/b41f28b3-cb74-439e-b73a-297c327e251d" />
2. <br> <img width="900" height="506" alt="image" src="https://github.com/user-attachments/assets/a39f3aff-7559-441c-8e5a-ef3913a9cb4f" />
3. <br> <img width="900" height="506" alt="image" src="https://github.com/user-attachments/assets/1d4d249d-be3e-48a1-b30f-a8b84010d96c" />
4. <br> <img width="900" height="506" alt="image" src="https://github.com/user-attachments/assets/40457522-9c00-485c-b03e-656690611eca" />

---

## Experimentation
### Implementation
All experiments are conducted in an inference-only setting on a single NVIDIA A100 GPU (40GB). The proposed framework integrates medical visual grounding directly into the autoregressive decoding process of the backbone medical VLM, without modifying model parameters or requiring additional training. For domain-specific visual–semantic alignment, we employ BioMedCLIP (microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224) as the image–text similarity encoder. BioMedCLIP, pretrained on biomedical image–text pairs, provides clinically relevant alignment signals that are more suitable for medical reasoning than generic vision–language encoders. During generation, candidate tokens are dynamically reweighted based on their contextual visual consistency using the proposed sliding-window calibration strategy. The following hyperparameters are used during inference:
* visual-top-k: 30
* window-size: 8
* batch-size: 16
* temperature: 0.0 (greedy decoding

**The visual-top-k parameter limits visual alignment evaluation to the top-30 candidate tokens at each decoding step, balancing computational efficiency and grounding precision. A sliding window of size 8 preserves short-term contextual coherence during reasoning. All remaining preprocessing, prompting templates, and evaluation procedures follow standard medical VQA protocols.** the parameters are as mentioned in

```bash
bash run.sh
```

### Datasets
We evaluate the proposed framework on the open-access portion of the OmniMedVQA benchmark, a large-scale medical vision question answering dataset comprising 82,059 images and 88,996 question–answer pairs. OmniMedVQA spans eight imaging modalities, including CT (15,808), MRI (31,877), X-Ray (7,916), Ultrasound (10,991), Dermoscopy (6,679), Fundus (5,398), OCT (4,646), and Microscopy (5,680), thereby covering a broad spectrum of anatomical, pathological, and cellular imaging scenarios. In addition to modality diversity, the dataset is organized into five clinically meaningful question categories: Anatomy Identification (16,448), Disease Diagnosis (55,387), Lesion Grading (2,098), Modality Recognition (11,565), and Other Biological Attributes (3,498). This structured taxonomy enables systematic evaluation across both cross-modality and cross-task reasoning settings. Following the official benchmark protocol, the dataset is partitioned into training and testing splits using an 80–20 ratio. In our experiments, we strictly operate in an inference-only regime and utilize only the official testing split for evaluation. No additional fine-tuning or task-specific adaptation is performed on OmniMedVQA, ensuring that performance gains arise solely from the proposed inference-time visual grounding mechanism.

### Summary of Results 
We comprehensively evaluate the performance of the proposed I<sup>2</sup>Med model across eight distinct medical imaging modalities. Comparison of performance across eight medical modalities against baseline models in Table 1. Comparison of our proposed models with medical VLMs baselines on five medical VQA tasks across five clinical reasoning types, evaluated under general-purpose zero-shot, medical zero-shot, and fine-tuned models in Table 2. The quantitative results are summarized in Table S.1. Additionally, we assess the effectiveness of I<sup>2</sup>Med on five clinical reasoning tasks to examine its generalization capability beyond imaging-based evaluation. The corresponding results are reported in Table S.2.

<p align="center">
  <img
    width="991"
    height="535"
    alt="image"
    src="https://github.com/user-attachments/assets/646079d3-a1b2-43af-8528-3cf7e19a36e1"
  />
</p>
<p align="center">
  <img
    width="967"
    height="516"
    alt="image"
    src="https://github.com/user-attachments/assets/558a31c2-34f1-41b4-ada8-ba6501c32c2e"
  />
</p>
<p align="center">
  <img
    width="836"
    height="306"
    alt="image"
    src="https://github.com/user-attachments/assets/ce4ad0f5-6ba4-4d8e-9629-5cd4bda219f5"
  />
</p>

<p align="center">
  <img
    width="840"
    height="215"
    alt="image"
    src="https://github.com/user-attachments/assets/ca0f9f32-e98b-4da9-aa50-e1dffa9b283d"
  />
</p>


## Ablation Study
Here is how Window size affects the sensitivity of Logits calibration.
<br>

**Table: Window size vs Accuracy on CT scan Vs CT scan**
| window size| Accuracy  |
| --- | --- | 
| 8| 61.09%|
| 12 | 59.75% |
| 16 | 60.88% |

---

## 📬 Contact

If you have any questions, encounter bugs, experience issues while reproducing the results, or have suggestions for improving the code or the framework, please feel free to contact any of the authors below.

| Author | Email |
|--------|-------|
| **M. Sajid** | phd2101241003@iiti.ac.in; sajid.mathml@gmail.com |
| **Shreeyut Maheshwari** | me230003071@iiti.ac.in |
| **Akshat Mishra** | akshatm430@gmail.com |
| **Ritik Mishra** | phd2301241003@iiti.ac.in |
| **Prof. Mohammad Tanveer** | mtanveer@iiti.ac.in |

Alternatively, you may open an **Issue** or submit a **Pull Request** on GitHub if you discover bugs or would like to contribute to the project.

If you find this repository useful in your research, please consider giving it a ⭐ and citing our paper.

---

## Acknowledgements

The work of Md Sajid is supported by the Council of Scientific and Industrial Research (CSIR), New Delhi, for providing fellowship under the Grant 09/1022(13847)/2022-EMR-I. This work is supported by PraxiaTech Private Limited. Further, this work is supported by IITI DRISHTI CPS Foundation under the National Mission on Interdisciplinary Cyber Physical System (NM-ICPS) of the Department of Science and Technology, Government of India.

---
