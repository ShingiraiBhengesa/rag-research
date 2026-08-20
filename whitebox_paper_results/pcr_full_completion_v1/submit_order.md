# Paper PCR Rerun Submit Order

These jobs write only under this results tree.

1. Freeze canonical raw samples.

   sbatch /home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/sbatch/prepare_samples.sbatch

2. Run clean baselines for every model/dataset.

   sbatch /home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/sbatch/pcr_baseline_qwen7b_hotpot.sbatch
   sbatch /home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/sbatch/pcr_baseline_qwen7b_2wiki.sbatch
   sbatch /home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/sbatch/pcr_baseline_qwen7b_musique.sbatch
   sbatch /home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/sbatch/pcr_baseline_qwen7b_multihoprag.sbatch
   sbatch /home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/sbatch/pcr_baseline_qwen14b_hotpot.sbatch
   sbatch /home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/sbatch/pcr_baseline_qwen14b_2wiki.sbatch
   sbatch /home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/sbatch/pcr_baseline_qwen14b_musique.sbatch
   sbatch /home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/sbatch/pcr_baseline_qwen14b_multihoprag.sbatch
   sbatch /home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/sbatch/pcr_baseline_llama8b_hotpot.sbatch
   sbatch /home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/sbatch/pcr_baseline_llama8b_2wiki.sbatch
   sbatch /home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/sbatch/pcr_baseline_llama8b_musique.sbatch
   sbatch /home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/sbatch/pcr_baseline_llama8b_multihoprag.sbatch
   sbatch /home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/sbatch/pcr_baseline_gemma27b_hotpot.sbatch
   sbatch /home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/sbatch/pcr_baseline_gemma27b_2wiki.sbatch
   sbatch /home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/sbatch/pcr_baseline_gemma27b_musique.sbatch
   sbatch /home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/sbatch/pcr_baseline_gemma27b_multihoprag.sbatch
   sbatch /home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/sbatch/pcr_baseline_gptoss20b_hotpot.sbatch
   sbatch /home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/sbatch/pcr_baseline_gptoss20b_2wiki.sbatch
   sbatch /home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/sbatch/pcr_baseline_gptoss20b_musique.sbatch
   sbatch /home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/sbatch/pcr_baseline_gptoss20b_multihoprag.sbatch

3. Build global eligible manifests from the baseline intersection.

   sbatch /home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/sbatch/build_global_eligible.sbatch

4. Run WGA attacks using the global eligible manifests.

   sbatch /home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/sbatch/pcr_wga_qwen7b_hotpot.sbatch
   sbatch /home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/sbatch/pcr_wga_qwen7b_2wiki.sbatch
   sbatch /home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/sbatch/pcr_wga_qwen7b_musique.sbatch
   sbatch /home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/sbatch/pcr_wga_qwen7b_multihoprag.sbatch
   sbatch /home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/sbatch/pcr_wga_qwen14b_hotpot.sbatch
   sbatch /home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/sbatch/pcr_wga_qwen14b_2wiki.sbatch
   sbatch /home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/sbatch/pcr_wga_qwen14b_musique.sbatch
   sbatch /home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/sbatch/pcr_wga_qwen14b_multihoprag.sbatch
   sbatch /home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/sbatch/pcr_wga_llama8b_hotpot.sbatch
   sbatch /home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/sbatch/pcr_wga_llama8b_2wiki.sbatch
   sbatch /home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/sbatch/pcr_wga_llama8b_musique.sbatch
   sbatch /home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/sbatch/pcr_wga_llama8b_multihoprag.sbatch
   sbatch /home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/sbatch/pcr_wga_gemma27b_hotpot.sbatch
   sbatch /home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/sbatch/pcr_wga_gemma27b_2wiki.sbatch
   sbatch /home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/sbatch/pcr_wga_gemma27b_musique.sbatch
   sbatch /home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/sbatch/pcr_wga_gemma27b_multihoprag.sbatch
   sbatch /home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/sbatch/pcr_wga_gptoss20b_hotpot.sbatch
   sbatch /home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/sbatch/pcr_wga_gptoss20b_2wiki.sbatch
   sbatch /home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/sbatch/pcr_wga_gptoss20b_musique.sbatch
   sbatch /home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/sbatch/pcr_wga_gptoss20b_multihoprag.sbatch

5. Run FactOnly and CueOnly after each matching WGA job.

   sbatch /home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/sbatch/pcr_factonly_qwen7b_hotpot.sbatch
   sbatch /home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/sbatch/pcr_cueonly_qwen7b_hotpot.sbatch
   sbatch /home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/sbatch/pcr_factonly_qwen7b_2wiki.sbatch
   sbatch /home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/sbatch/pcr_cueonly_qwen7b_2wiki.sbatch
   sbatch /home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/sbatch/pcr_factonly_qwen7b_musique.sbatch
   sbatch /home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/sbatch/pcr_cueonly_qwen7b_musique.sbatch
   sbatch /home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/sbatch/pcr_factonly_qwen7b_multihoprag.sbatch
   sbatch /home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/sbatch/pcr_cueonly_qwen7b_multihoprag.sbatch
   sbatch /home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/sbatch/pcr_factonly_qwen14b_hotpot.sbatch
   sbatch /home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/sbatch/pcr_cueonly_qwen14b_hotpot.sbatch
   sbatch /home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/sbatch/pcr_factonly_qwen14b_2wiki.sbatch
   sbatch /home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/sbatch/pcr_cueonly_qwen14b_2wiki.sbatch
   sbatch /home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/sbatch/pcr_factonly_qwen14b_musique.sbatch
   sbatch /home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/sbatch/pcr_cueonly_qwen14b_musique.sbatch
   sbatch /home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/sbatch/pcr_factonly_qwen14b_multihoprag.sbatch
   sbatch /home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/sbatch/pcr_cueonly_qwen14b_multihoprag.sbatch
   sbatch /home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/sbatch/pcr_factonly_llama8b_hotpot.sbatch
   sbatch /home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/sbatch/pcr_cueonly_llama8b_hotpot.sbatch
   sbatch /home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/sbatch/pcr_factonly_llama8b_2wiki.sbatch
   sbatch /home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/sbatch/pcr_cueonly_llama8b_2wiki.sbatch
   sbatch /home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/sbatch/pcr_factonly_llama8b_musique.sbatch
   sbatch /home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/sbatch/pcr_cueonly_llama8b_musique.sbatch
   sbatch /home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/sbatch/pcr_factonly_llama8b_multihoprag.sbatch
   sbatch /home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/sbatch/pcr_cueonly_llama8b_multihoprag.sbatch
   sbatch /home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/sbatch/pcr_factonly_gemma27b_hotpot.sbatch
   sbatch /home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/sbatch/pcr_cueonly_gemma27b_hotpot.sbatch
   sbatch /home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/sbatch/pcr_factonly_gemma27b_2wiki.sbatch
   sbatch /home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/sbatch/pcr_cueonly_gemma27b_2wiki.sbatch
   sbatch /home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/sbatch/pcr_factonly_gemma27b_musique.sbatch
   sbatch /home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/sbatch/pcr_cueonly_gemma27b_musique.sbatch
   sbatch /home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/sbatch/pcr_factonly_gemma27b_multihoprag.sbatch
   sbatch /home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/sbatch/pcr_cueonly_gemma27b_multihoprag.sbatch
   sbatch /home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/sbatch/pcr_factonly_gptoss20b_hotpot.sbatch
   sbatch /home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/sbatch/pcr_cueonly_gptoss20b_hotpot.sbatch
   sbatch /home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/sbatch/pcr_factonly_gptoss20b_2wiki.sbatch
   sbatch /home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/sbatch/pcr_cueonly_gptoss20b_2wiki.sbatch
   sbatch /home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/sbatch/pcr_factonly_gptoss20b_musique.sbatch
   sbatch /home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/sbatch/pcr_cueonly_gptoss20b_musique.sbatch
   sbatch /home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/sbatch/pcr_factonly_gptoss20b_multihoprag.sbatch
   sbatch /home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/sbatch/pcr_cueonly_gptoss20b_multihoprag.sbatch

The generated submit helper wires these phases with afterok dependencies:

   python /home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/submit_pipeline.py
