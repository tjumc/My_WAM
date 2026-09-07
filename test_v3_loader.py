"""
Test script for LeRobot v3.0 format support.
Tests basic dataset loading functionality.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_v3_dataset_loading():
    """Test loading a v3.0 format dataset."""
    from fastwam.datasets.lerobot.lerobot.lerobot_dataset import LeRobotDatasetMetadata, LeRobotDataset
    
    # Test dataset path (adjust if needed)
    dataset_dir = Path("data/midea_500h_v30")
    
    print(f"Testing v3.0 dataset loading from: {dataset_dir}")
    print("=" * 60)
    
    # Find first valid subdataset
    subdatasets = []
    for subdir in dataset_dir.iterdir():
        if subdir.is_dir() and (subdir / "meta" / "info.json").exists():
            subdatasets.append(subdir)
    
    if not subdatasets:
        print(f"❌ No valid subdatasets found in {dataset_dir}")
        print("   Each subdataset should have meta/info.json")
        return False
    
    test_dataset = subdatasets[0]
    print(f"✓ Found {len(subdatasets)} subdatasets")
    print(f"✓ Testing with: {test_dataset.name}")
    print()
    
    # Test 1: Load metadata
    print("Test 1: Loading metadata...")
    try:
        meta = LeRobotDatasetMetadata(
            repo_id=str(test_dataset),
            root=test_dataset
        )
        print(f"  ✓ Version: {meta._version}")
        print(f"  ✓ FPS: {meta.fps}")
        print(f"  ✓ Total episodes: {meta.total_episodes}")
        print(f"  ✓ Total frames: {meta.total_frames}")
        print(f"  ✓ Features: {list(meta.features.keys())}")
    except Exception as e:
        print(f"  ❌ Failed to load metadata: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print()
    
    # Test 2: Load dataset
    print("Test 2: Loading dataset (first 2 episodes)...")
    try:
        dataset = LeRobotDataset(
            repo_id=str(test_dataset),
            root=test_dataset,
            episodes=[0, 1] if meta.total_episodes > 1 else [0]
        )
        print(f"  ✓ Dataset length: {len(dataset)}")
        print(f"  ✓ Number of episodes: {dataset.num_episodes}")
    except Exception as e:
        print(f"  ❌ Failed to load dataset: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print()
    
    # Test 3: Get a sample
    print("Test 3: Getting first sample...")
    try:
        sample = dataset[0]
        print(f"  ✓ Sample keys: {list(sample.keys())}")
        print(f"  ✓ Episode index: {sample['episode_index'].item()}")
        print(f"  ✓ Frame index: {sample['frame_index'].item()}")
        print(f"  ✓ Task: {sample.get('task', 'N/A')}")
    except Exception as e:
        print(f"  ❌ Failed to get sample: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print()
    print("=" * 60)
    print("✅ All tests passed!")
    return True


if __name__ == "__main__":
    success = test_v3_dataset_loading()
    sys.exit(0 if success else 1)
