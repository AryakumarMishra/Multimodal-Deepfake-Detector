# debug_file.py
import torch
import os

# --- IMPORTANT: Adjust this path to be correct relative to where you run the script ---
MODEL_PATH = "backend/models/best_photo_detector.pth"

print("-" * 50)
if not os.path.exists(MODEL_PATH):
    print(f"ERROR: File not found at '{MODEL_PATH}'")
else:
    try:
        print(f"Attempting to load: {MODEL_PATH}")
        state_dict = torch.load(MODEL_PATH, map_location='cpu')

        if isinstance(state_dict, dict):
            print("Successfully loaded a dictionary (state_dict).")
            key_count = len(state_dict.keys())
            print(f"Number of keys found: {key_count}")

            if key_count == 0:
                print("\n>>> DIAGNOSIS: The state dictionary is EMPTY. This is the cause of your IndexError.")
            else:
                print("\n>>> DIAGNOSIS: The file seems OK. First 5 keys are:")
                for i, key in enumerate(list(state_dict.keys())[:5]):
                    print(f"  {i+1}: {key}")

        else:
            print(f"\n>>> DIAGNOSIS: The file is not a state_dict. It's a {type(state_dict)} object.")

    except Exception as e:
        print(f"\n>>> DIAGNOSIS: An error occurred while loading the file. It may be corrupt. Error: {e}")

print("-" * 50)