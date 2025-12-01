import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

DATA_PATH = Path(__file__).parent / "list.txt"
OUT_DIR = DATA_PATH.parent

# read space-separated lines, ignore comment lines starting with '#'
df = pd.read_csv(
    DATA_PATH,
    sep=" ",
    # header=None,
    names=["image", "class_id", "species", "breed_id"],
    dtype={"image": str, "class_id": int, "species": int, "breed_id": int},
)

# map species/breed codes to names
species_map = {1: "Cat", 2: "Dog"}
df["species_name"] = df["species"].map(species_map).fillna(df["species"].astype(str))

# counts per species
species_counts = df["species_name"].value_counts().sort_index()

plt.figure(figsize=(6, 4))
sns.barplot(x=species_counts.index, y=species_counts.values)
plt.title("Images per Species")
plt.xlabel("Species")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(OUT_DIR / "images_per_species.png")
plt.show()

# counts per class (class_id)
df["breed_name"] = df["image"].str.split("_").str[:-1].str.join("_")
# counts per class (class_id)
class_counts = df["class_id"].value_counts().sort_index()
# map class_id to breed_name (use first occurrence for each class)
breed_names = df.groupby("class_id")["breed_name"].first().to_dict()
x_labels = [f"{class_id} ({breed_names.get(class_id, 'Unknown')})" for class_id in class_counts.index]

plt.figure(figsize=(12, 5))
sns.barplot(x=x_labels, y=class_counts.values)
plt.title("Images per Breed")
plt.xlabel("Breed")
plt.ylabel("Count")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(OUT_DIR / "images_per_class.png")
plt.show()