#!/usr/bin/env python3
"""Example: Train student model with DPO on preference pairs.

This script demonstrates:
1. Creating or loading DPO preference pairs
2. Running DPO training from an SFT checkpoint
3. Registering the DPO adapter with lineage tracking

Usage:
    python examples/train_student_dpo.py \
        --sft-model outputs/student_sft/final \
        --dpo-data data/exports/dpo_pairs.jsonl

Requirements:
    - GPU with 8GB+ VRAM
    - SFT checkpoint from previous training
    - DPO preference pairs (or use --use-synthetic to create test data)
"""

import argparse
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def create_synthetic_dpo_pairs(output_path: Path, num_pairs: int = 20) -> Path:
    """Create synthetic DPO preference pairs for testing."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pairs = []
    for i in range(num_pairs):
        # Create a pair with chosen (better) and rejected (worse) responses
        pair = {
            "prompt": f"Describe the video segment {i}.",
            "chosen": "The person is carefully washing their hands with soap and water, "
            "ensuring thorough cleaning before handling food. This demonstrates "
            "proper hygiene practices in the kitchen environment.",
            "rejected": "Someone washing hands.",
            "video_id": f"video_{i:04d}",
            "metadata": {
                "chosen_score": 4.5,
                "rejected_score": 2.0,
                "judge": "synthetic",
            },
        }
        pairs.append(pair)

    with open(output_path, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    print(f"Created {num_pairs} synthetic DPO pairs: {output_path}")
    return output_path


def create_dpo_pairs_from_annotations(
    annotation_store_path: Path,
    output_path: Path,
) -> Path:
    """Create DPO pairs from annotation store with multiple teacher responses."""
    from dvas.data.storage import AnnotationStore

    output_path.parent.mkdir(parents=True, exist_ok=True)

    store = AnnotationStore(annotation_store_path)

    # Group annotations by video
    video_annotations = {}
    for ann in store.load_all():
        video_id = ann.video_id
        if video_id not in video_annotations:
            video_annotations[video_id] = []
        video_annotations[video_id].append(ann)

    # Create pairs from rankings
    pairs = []
    for video_id, annotations in video_annotations.items():
        if len(annotations) < 2:
            continue

        # Sort by quality score (could use LLM judge or other metric)
        # For now, use caption length as proxy
        ranked = sorted(
            annotations,
            key=lambda a: len(a.caption or ""),
            reverse=True,
        )

        # Generate pairs
        for i, winner in enumerate(ranked):
            for loser in ranked[i + 1 :]:
                pair = {
                    "prompt": "Describe the video in detail.",
                    "chosen": winner.caption or "",
                    "rejected": loser.caption or "",
                    "video_id": video_id,
                    "metadata": {
                        "winner_teacher": winner.model_type,
                        "loser_teacher": loser.model_type,
                    },
                }
                pairs.append(pair)

    with open(output_path, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    print(f"Created {len(pairs)} DPO pairs from annotations: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Train student model with DPO")
    parser.add_argument(
        "--sft-model",
        type=Path,
        required=True,
        help="Path to SFT model checkpoint",
    )
    parser.add_argument(
        "--dpo-data",
        type=Path,
        help="Path to DPO preference pairs JSONL",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/student_dpo"),
        help="Output directory",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.1,
        help="KL divergence coefficient (DPO beta)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size per device",
    )
    parser.add_argument(
        "--use-synthetic",
        action="store_true",
        help="Create synthetic DPO pairs for testing",
    )
    parser.add_argument(
        "--from-annotations",
        type=Path,
        help="Create DPO pairs from annotation store at given path",
    )
    parser.add_argument(
        "--register",
        action="store_true",
        help="Register trained adapter in registry",
    )
    parser.add_argument(
        "--registry-dir",
        type=Path,
        default=Path("outputs/model_registry"),
        help="Model registry directory",
    )
    parser.add_argument(
        "--sft-adapter-id",
        type=str,
        help="Registry ID of parent SFT adapter (for lineage tracking)",
    )
    parser.add_argument(
        "--data-version",
        type=str,
        default=None,
        help="Version identifier for DPO training data",
    )

    args = parser.parse_args()

    # Validate SFT model exists
    if not args.sft_model.exists():
        print(f"Error: SFT model not found: {args.sft_model}")
        print("Run SFT training first: python examples/train_student_sft.py")
        sys.exit(1)

    # Handle DPO data creation
    if args.use_synthetic:
        args.dpo_data = Path("data/exports/synthetic_dpo.jsonl")
        create_synthetic_dpo_pairs(args.dpo_data, num_pairs=20)
        args.data_version = args.data_version or "synthetic-v1"
    elif args.from_annotations:
        args.dpo_data = Path("data/exports/from_annotations_dpo.jsonl")
        create_dpo_pairs_from_annotations(args.from_annotations, args.dpo_data)
        args.data_version = args.data_version or "from-annotations-v1"

    if not args.dpo_data or not args.dpo_data.exists():
        print(f"Error: DPO data not found: {args.dpo_data}")
        print("Use --use-synthetic to create test data")
        sys.exit(1)

    print("=" * 60)
    print("STUDENT DPO TRAINING")
    print("=" * 60)
    print(f"SFT model: {args.sft_model}")
    print(f"DPO data: {args.dpo_data}")
    print(f"Output directory: {args.output_dir}")
    print(f"Beta: {args.beta}")
    print(f"Epochs: {args.epochs}")
    print("=" * 60)

    try:
        from dvas.models.student import DPOConfig, train_dpo
        from dvas.models.student.registry import LoRAAdapterRegistry

        # Build configuration
        config = DPOConfig()
        config.ref_model_path = args.sft_model
        config.data.train_data_path = args.dpo_data
        config.training.output_dir = args.output_dir
        config.training.beta = args.beta
        config.training.num_train_epochs = args.epochs
        config.data.batch_size = args.batch_size
        config.model.use_lora = True
        config.model.load_in_4bit = True
        config.report_to = "none"

        # Run training
        print("\nStarting DPO training...")
        final_path = train_dpo(config)
        print(f"\nDPO training complete! Model saved to: {final_path}")

        # Register in registry if requested
        if args.register:
            print("\nRegistering DPO adapter in model registry...")
            registry = LoRAAdapterRegistry(args.registry_dir)

            adapter_id = registry.register_adapter(
                adapter_path=final_path,
                adapter_name="dvas_student_dpo",
                base_model=config.model.model_name_or_path,
                adapter_type="lora",
                training_data_path=args.dpo_data,
                training_config=config,
                parent_adapter_id=args.sft_adapter_id,
                metrics={"beta": args.beta, "epochs": args.epochs},
                tags=["dpo", "qwen2-vl", "preference-optimized"],
                description=f"DPO trained student model (beta={args.beta}, {args.epochs} epochs)",
                data_version=args.data_version,
            )

            print(f"Registered DPO adapter: {adapter_id}")

            # Show lineage
            if args.sft_adapter_id:
                print(f"  - Parent (SFT): {args.sft_adapter_id}")
                lineage = registry.get_adapter_lineage(adapter_id)
                print(f"  - Lineage depth: {len(lineage)} adapters")

            metadata = registry.get_metadata(adapter_id)
            if metadata:
                print(f"  - Training data hash: {metadata.training_data_hash}")
                print(f"  - Data version: {metadata.data_version}")

        print("\n" + "=" * 60)
        print("SUCCESS")
        print("=" * 60)
        print(f"Model location: {final_path}")
        if args.register:
            print(f"Registry location: {args.registry_dir}")
        print("\nNext steps:")
        print("  1. Run inference: python examples/inference_student.py")
        print("  2. Evaluate: python examples/eval_teacher_vs_student.py")
        print("  3. Deploy with fallback: python examples/deploy_with_fallback.py")

    except ImportError as e:
        print(f"\nError: Missing required dependencies: {e}")
        print("\nPlease install training dependencies:")
        print("  pip install torch transformers peft trl datasets bitsandbytes")
        sys.exit(1)
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
