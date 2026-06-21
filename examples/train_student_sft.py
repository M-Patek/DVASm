#!/usr/bin/env python3
"""Example: Train student model with SFT on mini dataset.

This script demonstrates:
1. Creating a mini training dataset
2. Running SFT training with LoRA
3. Registering the trained adapter

Usage:
    python examples/train_student_sft.py --data data/exports/train_llava.jsonl

Requirements:
    - GPU with 8GB+ VRAM (for 4-bit quantization)
    - Training data in LLaVA format
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def create_mini_dataset(output_path: Path, num_samples: int = 10) -> Path:
    """Create a minimal dataset for testing."""
    import json

    output_path.parent.mkdir(parents=True, exist_ok=True)

    samples = []
    for i in range(num_samples):
        sample = {
            "id": f"mini_{i:04d}",
            "video": f"/fake/video_{i}.mp4",
            "conversations": [
                {
                    "from": "human",
                    "value": "<video>\nDescribe the actions in this video segment.",
                },
                {
                    "from": "gpt",
                    "value": f"The person is performing action {i}: cutting vegetables, washing hands, and preparing ingredients.",
                },
            ],
        }
        samples.append(sample)

    with open(output_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"Created mini dataset with {num_samples} samples: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Train student model with SFT")
    parser.add_argument(
        "--data",
        type=Path,
        help="Path to training data (LLaVA format JSONL)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/student_sft"),
        help="Output directory for trained model",
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
        "--lora-r",
        type=int,
        default=64,
        help="LoRA rank",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=128,
        help="LoRA alpha",
    )
    parser.add_argument(
        "--use-mini",
        action="store_true",
        help="Create and use mini dataset for testing",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of samples for mini dataset",
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
        "--data-version",
        type=str,
        default=None,
        help="Version identifier for training data",
    )
    parser.add_argument(
        "--epic-split",
        type=str,
        default=None,
        help="EPIC-KITCHENS split used (e.g., 'train', 'val')",
    )

    args = parser.parse_args()

    # Handle mini dataset creation
    if args.use_mini:
        mini_path = Path("data/exports/mini_train.jsonl")
        args.data = create_mini_dataset(mini_path, args.num_samples)
        args.data_version = args.data_version or "mini-v1"

    if not args.data or not args.data.exists():
        print(f"Error: Training data not found: {args.data}")
        print("Use --use-mini to create a test dataset")
        sys.exit(1)

    print("=" * 60)
    print("STUDENT SFT TRAINING")
    print("=" * 60)
    print(f"Training data: {args.data}")
    print(f"Output directory: {args.output_dir}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"LoRA r: {args.lora_r}, alpha: {args.lora_alpha}")
    print("=" * 60)

    try:
        from dvas.models.student import SFTConfig, train_sft
        from dvas.models.student.registry import LoRAAdapterRegistry

        # Build configuration
        config = SFTConfig()
        config.data.train_data_path = args.data
        config.training.output_dir = args.output_dir
        config.training.num_train_epochs = args.epochs
        config.data.batch_size = args.batch_size
        config.model.lora_r = args.lora_r
        config.model.lora_alpha = args.lora_alpha
        config.model.use_lora = True
        config.model.load_in_4bit = True
        config.report_to = "none"

        # Run training
        print("\nStarting training...")
        final_path = train_sft(config)
        print(f"\nTraining complete! Model saved to: {final_path}")

        # Register in registry if requested
        if args.register:
            print("\nRegistering adapter in model registry...")
            registry = LoRAAdapterRegistry(args.registry_dir)

            adapter_id = registry.register_adapter(
                adapter_path=final_path,
                adapter_name="dvas_student_sft",
                base_model=config.model.model_name_or_path,
                adapter_type="lora",
                training_data_path=args.data,
                training_config=config,
                metrics={
                    "epochs": args.epochs,
                    "final_loss": 0.0,
                },  # Would get from actual training
                tags=["sft", "qwen2-vl", args.epic_split or "unknown"],
                description=f"SFT trained student model ({args.epochs} epochs)",
                data_version=args.data_version,
                epic_split=args.epic_split,
            )

            print(f"Registered adapter: {adapter_id}")

            # Verify registration
            metadata = registry.get_metadata(adapter_id)
            if metadata:
                print(f"  - Training data hash: {metadata.training_data_hash}")
                print(f"  - Data version: {metadata.data_version}")
                print(f"  - EPIC split: {metadata.epic_split}")

        print("\n" + "=" * 60)
        print("SUCCESS")
        print("=" * 60)
        print(f"Model location: {final_path}")
        if args.register:
            print(f"Registry location: {args.registry_dir}")
        print("\nNext steps:")
        print("  1. Run inference: python examples/inference_student.py")
        print("  2. Run DPO training: python examples/train_student_dpo.py")
        print("  3. Evaluate: python examples/eval_teacher_vs_student.py")

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
