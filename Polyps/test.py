import json

# Load the structured JSON
with open("kavsir_bboxes.json", "r") as f:
    data = json.load(f)

with open("kavsir_bboxes.json", "r") as f:
    data = json.load(f)

# Print contents (optional)
for image_id, info in data.items():
    print("Image ID:", image_id)
    print("Height:", info["height"])
    print("Width:", info["width"])
    
    for bbox in info["bbox"]:
        print("Label:", bbox["label"])
        print("Coordinates:", bbox["xmin"], bbox["ymin"], bbox["xmax"], bbox["ymax"])

# Save to a new file
with open("structured.json", "w") as f:
    json.dump(data, f, indent=4)