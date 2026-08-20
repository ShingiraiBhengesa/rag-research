#!/usr/bin/env python3
import argparse
import re
import subprocess

PREPARE = '/home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/sbatch/prepare_samples.sbatch'
BUILD = '/home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/sbatch/build_global_eligible.sbatch'
BASELINES = ['/home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/sbatch/pcr_baseline_qwen7b_hotpot.sbatch', '/home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/sbatch/pcr_baseline_qwen7b_2wiki.sbatch', '/home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/sbatch/pcr_baseline_qwen7b_musique.sbatch', '/home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/sbatch/pcr_baseline_qwen7b_multihoprag.sbatch', '/home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/sbatch/pcr_baseline_qwen14b_hotpot.sbatch', '/home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/sbatch/pcr_baseline_qwen14b_2wiki.sbatch', '/home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/sbatch/pcr_baseline_qwen14b_musique.sbatch', '/home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/sbatch/pcr_baseline_qwen14b_multihoprag.sbatch', '/home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/sbatch/pcr_baseline_llama8b_hotpot.sbatch', '/home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/sbatch/pcr_baseline_llama8b_2wiki.sbatch', '/home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/sbatch/pcr_baseline_llama8b_musique.sbatch', '/home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/sbatch/pcr_baseline_llama8b_multihoprag.sbatch', '/home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/sbatch/pcr_baseline_gemma27b_hotpot.sbatch', '/home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/sbatch/pcr_baseline_gemma27b_2wiki.sbatch', '/home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/sbatch/pcr_baseline_gemma27b_musique.sbatch', '/home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/sbatch/pcr_baseline_gemma27b_multihoprag.sbatch', '/home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/sbatch/pcr_baseline_gptoss20b_hotpot.sbatch', '/home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/sbatch/pcr_baseline_gptoss20b_2wiki.sbatch', '/home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/sbatch/pcr_baseline_gptoss20b_musique.sbatch', '/home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/sbatch/pcr_baseline_gptoss20b_multihoprag.sbatch']
WGA = {'qwen7b:hotpot': '/home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/sbatch/pcr_wga_qwen7b_hotpot.sbatch', 'qwen7b:2wiki': '/home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/sbatch/pcr_wga_qwen7b_2wiki.sbatch', 'qwen7b:musique': '/home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/sbatch/pcr_wga_qwen7b_musique.sbatch', 'qwen7b:multihoprag': '/home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/sbatch/pcr_wga_qwen7b_multihoprag.sbatch', 'qwen14b:hotpot': '/home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/sbatch/pcr_wga_qwen14b_hotpot.sbatch', 'qwen14b:2wiki': '/home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/sbatch/pcr_wga_qwen14b_2wiki.sbatch', 'qwen14b:musique': '/home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/sbatch/pcr_wga_qwen14b_musique.sbatch', 'qwen14b:multihoprag': '/home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/sbatch/pcr_wga_qwen14b_multihoprag.sbatch', 'llama8b:hotpot': '/home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/sbatch/pcr_wga_llama8b_hotpot.sbatch', 'llama8b:2wiki': '/home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/sbatch/pcr_wga_llama8b_2wiki.sbatch', 'llama8b:musique': '/home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/sbatch/pcr_wga_llama8b_musique.sbatch', 'llama8b:multihoprag': '/home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/sbatch/pcr_wga_llama8b_multihoprag.sbatch', 'gemma27b:hotpot': '/home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/sbatch/pcr_wga_gemma27b_hotpot.sbatch', 'gemma27b:2wiki': '/home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/sbatch/pcr_wga_gemma27b_2wiki.sbatch', 'gemma27b:musique': '/home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/sbatch/pcr_wga_gemma27b_musique.sbatch', 'gemma27b:multihoprag': '/home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/sbatch/pcr_wga_gemma27b_multihoprag.sbatch', 'gptoss20b:hotpot': '/home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/sbatch/pcr_wga_gptoss20b_hotpot.sbatch', 'gptoss20b:2wiki': '/home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/sbatch/pcr_wga_gptoss20b_2wiki.sbatch', 'gptoss20b:musique': '/home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/sbatch/pcr_wga_gptoss20b_musique.sbatch', 'gptoss20b:multihoprag': '/home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/sbatch/pcr_wga_gptoss20b_multihoprag.sbatch'}
FACTONLY = {'qwen7b:hotpot': '/home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/sbatch/pcr_factonly_qwen7b_hotpot.sbatch', 'qwen7b:2wiki': '/home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/sbatch/pcr_factonly_qwen7b_2wiki.sbatch', 'qwen7b:musique': '/home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/sbatch/pcr_factonly_qwen7b_musique.sbatch', 'qwen7b:multihoprag': '/home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/sbatch/pcr_factonly_qwen7b_multihoprag.sbatch', 'qwen14b:hotpot': '/home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/sbatch/pcr_factonly_qwen14b_hotpot.sbatch', 'qwen14b:2wiki': '/home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/sbatch/pcr_factonly_qwen14b_2wiki.sbatch', 'qwen14b:musique': '/home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/sbatch/pcr_factonly_qwen14b_musique.sbatch', 'qwen14b:multihoprag': '/home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/sbatch/pcr_factonly_qwen14b_multihoprag.sbatch', 'llama8b:hotpot': '/home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/sbatch/pcr_factonly_llama8b_hotpot.sbatch', 'llama8b:2wiki': '/home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/sbatch/pcr_factonly_llama8b_2wiki.sbatch', 'llama8b:musique': '/home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/sbatch/pcr_factonly_llama8b_musique.sbatch', 'llama8b:multihoprag': '/home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/sbatch/pcr_factonly_llama8b_multihoprag.sbatch', 'gemma27b:hotpot': '/home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/sbatch/pcr_factonly_gemma27b_hotpot.sbatch', 'gemma27b:2wiki': '/home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/sbatch/pcr_factonly_gemma27b_2wiki.sbatch', 'gemma27b:musique': '/home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/sbatch/pcr_factonly_gemma27b_musique.sbatch', 'gemma27b:multihoprag': '/home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/sbatch/pcr_factonly_gemma27b_multihoprag.sbatch', 'gptoss20b:hotpot': '/home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/sbatch/pcr_factonly_gptoss20b_hotpot.sbatch', 'gptoss20b:2wiki': '/home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/sbatch/pcr_factonly_gptoss20b_2wiki.sbatch', 'gptoss20b:musique': '/home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/sbatch/pcr_factonly_gptoss20b_musique.sbatch', 'gptoss20b:multihoprag': '/home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/sbatch/pcr_factonly_gptoss20b_multihoprag.sbatch'}
CUEONLY = {'qwen7b:hotpot': '/home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/sbatch/pcr_cueonly_qwen7b_hotpot.sbatch', 'qwen7b:2wiki': '/home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/sbatch/pcr_cueonly_qwen7b_2wiki.sbatch', 'qwen7b:musique': '/home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/sbatch/pcr_cueonly_qwen7b_musique.sbatch', 'qwen7b:multihoprag': '/home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/sbatch/pcr_cueonly_qwen7b_multihoprag.sbatch', 'qwen14b:hotpot': '/home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/sbatch/pcr_cueonly_qwen14b_hotpot.sbatch', 'qwen14b:2wiki': '/home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/sbatch/pcr_cueonly_qwen14b_2wiki.sbatch', 'qwen14b:musique': '/home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/sbatch/pcr_cueonly_qwen14b_musique.sbatch', 'qwen14b:multihoprag': '/home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/sbatch/pcr_cueonly_qwen14b_multihoprag.sbatch', 'llama8b:hotpot': '/home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/sbatch/pcr_cueonly_llama8b_hotpot.sbatch', 'llama8b:2wiki': '/home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/sbatch/pcr_cueonly_llama8b_2wiki.sbatch', 'llama8b:musique': '/home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/sbatch/pcr_cueonly_llama8b_musique.sbatch', 'llama8b:multihoprag': '/home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/sbatch/pcr_cueonly_llama8b_multihoprag.sbatch', 'gemma27b:hotpot': '/home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/sbatch/pcr_cueonly_gemma27b_hotpot.sbatch', 'gemma27b:2wiki': '/home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/sbatch/pcr_cueonly_gemma27b_2wiki.sbatch', 'gemma27b:musique': '/home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/sbatch/pcr_cueonly_gemma27b_musique.sbatch', 'gemma27b:multihoprag': '/home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/sbatch/pcr_cueonly_gemma27b_multihoprag.sbatch', 'gptoss20b:hotpot': '/home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/sbatch/pcr_cueonly_gptoss20b_hotpot.sbatch', 'gptoss20b:2wiki': '/home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/sbatch/pcr_cueonly_gptoss20b_2wiki.sbatch', 'gptoss20b:musique': '/home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/sbatch/pcr_cueonly_gptoss20b_musique.sbatch', 'gptoss20b:multihoprag': '/home/sbhengesa/rag-research/whitebox_paper_results/pcr_full_completion_v1/sbatch/pcr_cueonly_gptoss20b_multihoprag.sbatch'}


def submit(path, dependency=None, dry_run=False):
    cmd = ["sbatch"]
    if dependency:
        cmd.append(f"--dependency=afterok:{dependency}")
    cmd.append(path)
    print(" ".join(cmd), flush=True)
    if dry_run:
        submit.counter += 1
        return f"DRY{submit.counter}"
    out = subprocess.check_output(cmd, text=True)
    print(out.strip(), flush=True)
    match = re.search(r"Submitted batch job (\d+)", out)
    if not match:
        raise RuntimeError(f"Could not parse sbatch output: {out!r}")
    return match.group(1)


submit.counter = 0


def dep(ids):
    return ":".join(ids)


def main():
    parser = argparse.ArgumentParser(description="Submit the paper PCR rerun pipeline.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    prepare_id = submit(PREPARE, dry_run=args.dry_run)
    baseline_ids = [
        submit(path, dependency=prepare_id, dry_run=args.dry_run)
        for path in BASELINES
    ]
    build_id = submit(BUILD, dependency=dep(baseline_ids), dry_run=args.dry_run)

    wga_ids = {}
    for key, path in WGA.items():
        wga_ids[key] = submit(path, dependency=build_id, dry_run=args.dry_run)

    for key, path in FACTONLY.items():
        submit(path, dependency=wga_ids[key], dry_run=args.dry_run)
    for key, path in CUEONLY.items():
        submit(path, dependency=wga_ids[key], dry_run=args.dry_run)

    print("Pipeline submitted.")
    print(f"prepare={prepare_id} build_global_eligible={build_id}")


if __name__ == "__main__":
    main()
