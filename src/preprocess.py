import os
import json
import pandas as pd
from tqdm import tqdm

def create_manifest(base_path):
    # Matches your folder structure exactly
    root_dir = os.path.join(base_path, 'wYe7pBJ7-train', 'train')

    if not os.path.exists(root_dir):
        print(f"Error: {root_dir} not found!")
        return

    data = []
    scenarios = ['Scenario-A', 'Scenario-B']
    layouts = ['Brazilian', 'Mercosur']

    for scenario in scenarios:
        scenario_path = os.path.join(root_dir, scenario)
        if not os.path.exists(scenario_path): continue

        for layout in layouts:
            layout_path = os.path.join(scenario_path, layout)
            if not os.path.exists(layout_path): continue

            tracks = [t for t in os.listdir(layout_path) if os.path.isdir(os.path.join(layout_path, t))]

            for track_id in tqdm(tracks, desc=f"{scenario}/{layout}"):
                track_path = os.path.join(layout_path, track_id)
                json_path = os.path.join(track_path, 'annotations.json') # Fixed filename

                if not os.path.exists(json_path): continue

                with open(json_path, 'r', encoding='utf-8') as f:
                    meta = json.load(f)

                # Keep 5 images as pairs per track
                for i in range(1, 6):
                    lr_path = os.path.join(track_path, f"lr-00{i}.png")
                    hr_path = os.path.join(track_path, f"hr-00{i}.png")

                    if os.path.exists(lr_path):
                        data.append({
                            "scenario": scenario,
                            "layout": layout,
                            "track_id": track_id,
                            "plate_text": meta.get('plate_text', 'UNKNOWN'),
                            "lr_full_path": lr_path,
                            # Only include HR path if it's Scenario-A and file exists
                            "hr_full_path": hr_path if (scenario == "Scenario-A" and os.path.exists(hr_path)) else None
                        })

    df = pd.DataFrame(data)
    os.makedirs('data', exist_ok=True)
    out_path = 'data/training_manifest.csv'
    df.to_csv(out_path, index=False)
    print(f"\nSaved {len(df)} rows to {out_path}")

if __name__ == "__main__":
    create_manifest("data/raw")