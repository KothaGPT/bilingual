"""Test script to verify the dataset can be loaded correctly."""

from datasets import load_dataset


def test_dataset_loading():
    print("Testing dataset loading...")
    try:
        # Load the dataset
        dataset = load_dataset("KothaGPT/bilingual-corpus")

        # Print dataset information
        print("\nDataset loaded successfully!")
        print("\nDataset structure:", dataset)

        # Print features
        print("\nFeatures:", dataset["train"].features)

        # Print first example
        print("\nFirst training example:")
        print(dataset["train"][0])

        # Print dataset statistics
        print("\nDataset statistics:")
        for split in dataset:
            print(f"{split}: {len(dataset[split])} examples")

        return True

    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        return False


if __name__ == "__main__":
    test_dataset_loading()
