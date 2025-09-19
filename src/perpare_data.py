import os
import shutil
from sklearn.model_selection import train_test_split
import random

def organize_solar_data(source_dir, data_dir, train_pct=0.8, val_pct=0.1, test_pct=0.1):
    """
    Organize solar panel images with specific percentage splits

    Args:
        source_dir: Directory containing all your images
        data_dir: Base directory for organized data
        train_pct: Percentage for training set (0.0-1.0)
        val_pct: Percentage for validation set (0.0-1.0)
        test_pct: Percentage for test set (0.0-1.0)
    """

    # Validate percentages
    total_pct = train_pct + val_pct + test_pct
    if abs(total_pct - 1.0) > 0.01:
        raise ValueError(f"Percentages must sum to 1.0, got {total_pct}")

    print(f"Using split: Train={train_pct*100:.0f}%, Val={val_pct*100:.0f}%, Test={test_pct*100:.0f}%")

    # Create main directories
    os.makedirs(data_dir, exist_ok=True)

    # Process solar images
    solar_source = os.path.join(source_dir, 'with_panels')
    if os.path.exists(solar_source):
        solar_files = [f for f in os.listdir(solar_source)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        random.shuffle(solar_files)  # Shuffle for random split

        # Calculate split sizes
        n_solar = len(solar_files)
        n_train_solar = int(n_solar * train_pct)
        n_val_solar = int(n_solar * val_pct)
        n_test_solar = n_solar - n_train_solar - n_val_solar

        # Split files
        train_solar = solar_files[:n_train_solar]
        val_solar = solar_files[n_train_solar:n_train_solar + n_val_solar]
        test_solar = solar_files[n_train_solar + n_val_solar:]

        print(f"\n🌞 Solar Images: {n_solar} total")
        print(f"   → Train: {len(train_solar)} ({len(train_solar)/n_solar*100:.1f}%)")
        print(f"   → Val:   {len(val_solar)} ({len(val_solar)/n_solar*100:.1f}%)")
        print(f"   → Test:  {len(test_solar)} ({len(test_solar)/n_solar*100:.1f}%)")

        # Create and populate directories
        for split, files in [('train', train_solar), ('val', val_solar), ('test', test_solar)]:
            split_dir = os.path.join(data_dir, split, 'solar')
            os.makedirs(split_dir, exist_ok=True)

            for file in files:
                shutil.copy(os.path.join(solar_source, file), os.path.join(split_dir, file))

    # Process no_solar images
    nosolar_source = os.path.join(source_dir, 'no_panels')
    if os.path.exists(nosolar_source):
        nosolar_files = [f for f in os.listdir(nosolar_source)
                         if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        random.shuffle(nosolar_files)

        # Calculate split sizes
        n_nosolar = len(nosolar_files)
        n_train_nosolar = int(n_nosolar * train_pct)
        n_val_nosolar = int(n_nosolar * val_pct)
        n_test_nosolar = n_nosolar - n_train_nosolar - n_val_nosolar

        # Split files
        train_nosolar = nosolar_files[:n_train_nosolar]
        val_nosolar = nosolar_files[n_train_nosolar:n_train_nosolar + n_val_nosolar]
        test_nosolar = nosolar_files[n_train_nosolar + n_val_nosolar:]

        print(f"\n🏠 No Solar Images: {n_nosolar} total")
        print(f"   → Train: {len(train_nosolar)} ({len(train_nosolar)/n_nosolar*100:.1f}%)")
        print(f"   → Val:   {len(val_nosolar)} ({len(val_nosolar)/n_nosolar*100:.1f}%)")
        print(f"   → Test:  {len(test_nosolar)} ({len(test_nosolar)/n_nosolar*100:.1f}%)")

        # Create and populate directories
        for split, files in [('train', train_nosolar), ('val', val_nosolar), ('test', test_nosolar)]:
            split_dir = os.path.join(data_dir, split, 'no_solar')
            os.makedirs(split_dir, exist_ok=True)

            for file in files:
                shutil.copy(os.path.join(nosolar_source, file), os.path.join(split_dir, file))

    print(f"\n✅ Data organized successfully in '{data_dir}'!")

# Different split strategies
def create_different_splits():
    """Example usage with different split strategies"""

    source_directory = "../data/images"

    # Strategy 1: Standard 80/10/10
    print("=== STRATEGY 1: Standard Split (80/10/10) ===")
    organize_solar_data(source_directory, "data_standard", train_pct=0.8, val_pct=0.1, test_pct=0.1)

    # Strategy 2: More validation data (75/15/10)
    print("\n=== STRATEGY 2: Validation-Heavy (75/15/10) ===")
    organize_solar_data(source_directory, "data_valheavy", train_pct=0.75, val_pct=0.15, test_pct=0.1)

    # Strategy 3: Smaller test set (85/10/5)
    print("\n=== STRATEGY 3: Train-Heavy (85/10/5) ===")
    organize_solar_data(source_directory, "data_trainheavy", train_pct=0.85, val_pct=0.1, test_pct=0.05)

# Usage
if __name__ == "__main__":
    source_directory = "../data/images"
    organize_solar_data(source_directory, "data_split_standard", train_pct=0.8, val_pct=0.1, test_pct=0.1)