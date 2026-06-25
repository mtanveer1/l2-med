
## To Run 
1. Download the model weights of the VLM and BioMedCLIP
2. Set the parameters in run.sh
3. Use the following command
```bash
bash run.sh
```
## Interpretability
1. <img width="900" height="506" alt="image" src="https://github.com/user-attachments/assets/b41f28b3-cb74-439e-b73a-297c327e251d" />
2. <img width="900" height="506" alt="image" src="https://github.com/user-attachments/assets/a39f3aff-7559-441c-8e5a-ef3913a9cb4f" />
3. <img width="900" height="506" alt="image" src="https://github.com/user-attachments/assets/1d4d249d-be3e-48a1-b30f-a8b84010d96c" />
4. <img width="900" height="506" alt="image" src="https://github.com/user-attachments/assets/40457522-9c00-485c-b03e-656690611eca" />

## Experimentation
For reducing inferece overhead only top - 30 tokens are selected at each decoding step 
the parameters are as mentioned in 
```bash
bash run.sh
```

## Ablation Study
Window sensitivity of Logits calliberation

| window size| Accuracy  |
| --- | --- | 
| 8| 61.09%|
| 12 | 59.75% |
| 16 | 60.88% |
