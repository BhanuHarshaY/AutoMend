"""
captum_analysis.py

Model sensitivity analysis using Captum Integrated Gradients.

Traces back which tokenized infrastructure metrics (e.g., [MEM_9], [TMPL_402])
caused the model to fire a specific anomaly alert. This helps operators
understand *why* the model flagged a particular 5-minute window.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from loguru import logger


def run_integrated_gradients(
    model,
    tokenizer,
    dataset,
    int_to_token: Dict[int, str],
    n_samples: int = 200,
    n_steps: int = 50,
    device: str = "cpu",
    target_classes: Optional[List[int]] = None,
) -> Dict:
    """
    Run Integrated Gradients attribution on a subset of samples.

    Args:
        model: Trained RoBERTa model.
        tokenizer: Tokenizer with custom MLOps tokens.
        dataset: TrackADataset instance.
        int_to_token: Raw int ID -> token string mapping.
        n_samples: Number of samples to analyze.
        n_steps: Integration steps for IG.
        device: Torch device.
        target_classes: If set, only analyze samples predicted as these classes.

    Returns:
        Dict with per-token attribution summaries.
    """
    from captum.attr import IntegratedGradients, visualization

    model.to(device)
    model.eval()
    model.zero_grad()

    embeddings = model.get_input_embeddings()

    def forward_fn(input_embeds, attention_mask):
        outputs = model(inputs_embeds=input_embeds, attention_mask=attention_mask)
        return outputs.logits

    ig = IntegratedGradients(forward_fn)

    token_attributions: Dict[str, List[float]] = defaultdict(list)
    sample_results = []
    analyzed = 0

    indices = list(range(min(n_samples, len(dataset))))

    for idx in indices:
        item = dataset[idx]
        input_ids = item["input_ids"].unsqueeze(0).to(device)
        attention_mask = item["attention_mask"].unsqueeze(0).to(device)
        true_label = item["labels"].item()

        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            pred_label = logits.argmax(dim=-1).item()

        if target_classes and pred_label not in target_classes:
            continue

        input_embeds = embeddings(input_ids)
        baseline = torch.zeros_like(input_embeds)

        attrs = ig.attribute(
            input_embeds,
            baselines=baseline,
            additional_forward_args=(attention_mask,),
            target=pred_label,
            n_steps=n_steps,
        )

        attr_scores = attrs.squeeze(0).sum(dim=-1).cpu().numpy()
        tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0).cpu().tolist())

        active_mask = attention_mask.squeeze(0).cpu().numpy().astype(bool)
        top_indices = np.argsort(np.abs(attr_scores * active_mask))[-10:][::-1]

        top_tokens = []
        for ti in top_indices:
            tok = tokens[ti]
            score = float(attr_scores[ti])
            token_attributions[tok].append(score)
            top_tokens.append({"token": tok, "attribution": score, "position": int(ti)})

        sample_results.append({
            "index": idx,
            "true_label": true_label,
            "pred_label": pred_label,
            "top_tokens": top_tokens,
        })
        analyzed += 1

    global_importance = {}
    for tok, scores in token_attributions.items():
        global_importance[tok] = {
            "mean_attribution": float(np.mean(scores)),
            "abs_mean": float(np.mean(np.abs(scores))),
            "count": len(scores),
        }

    sorted_tokens = sorted(
        global_importance.items(), key=lambda x: x[1]["abs_mean"], reverse=True
    )

    logger.info(f"Analyzed {analyzed} samples. Top influential tokens:")
    for tok, info in sorted_tokens[:15]:
        logger.info(f"  {tok}: abs_mean={info['abs_mean']:.4f} (seen {info['count']}x)")

    return {
        "n_analyzed": analyzed,
        "global_token_importance": dict(sorted_tokens),
        "sample_attributions": sample_results,
    }


def save_sensitivity_report(results: Dict, output_dir: str | Path) -> Path:
    """Save the sensitivity analysis results to JSON."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    out_path = output_dir / "sensitivity_report.json"

    serializable = {
        "n_analyzed": results["n_analyzed"],
        "top_50_tokens": dict(list(results["global_token_importance"].items())[:50]),
        "sample_count": len(results["sample_attributions"]),
    }
    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2)

    full_path = output_dir / "sensitivity_full.json"
    with open(full_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"Sensitivity report -> {out_path}")
    return out_path
