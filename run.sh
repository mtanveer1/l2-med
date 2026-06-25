python main.py --prompt-path test.json \ 
--output save_results.jsonl \
--clip-model-path microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224 \ 
--max-new-tokens 256 \
--visual-top-k 30 \
--window-size 8 \
--batch-size 16 \
--model-path MedR1_VQA
