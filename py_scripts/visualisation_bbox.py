import csv
import json
import os
import cv2

# Répertoire contenant les images
image_dir = "datasets/YOLO-training/images200"  # Chemin des images

# Parcourir le fichier CSV
with open(csv_file, "r") as file:
    reader = csv.DictReader(file)
    for row in reader:
        filename = row["filename"]
        shape_attributes = row["region_shape_attributes"]

        # Extraire les coordonnées du polygone
        shape_data = json.loads(shape_attributes)
        all_points_x = shape_data["all_points_x"]
        all_points_y = shape_data["all_points_y"]

        # Calculer les coordonnées de la boîte englobante
        x_min, x_max = min(all_points_x), max(all_points_x)
        y_min, y_max = min(all_points_y), max(all_points_y)

        # Charger les dimensions de l'image
        image_path = os.path.join(image_dir, filename)
        image = cv2.imread(image_path)
        height, width, _ = image.shape

        # Calculer les valeurs normalisées pour YOLO
        x_center = ((x_min + x_max) / 2) / width
        y_center = ((y_min + y_max) / 2) / height
        box_width = (x_max - x_min) / width
        box_height = (y_max - y_min) / height

        # Sauvegarder les annotations YOLO
        annotation_file = os.path.join(output_dir, filename.replace(".jpg", ".txt"))
        with open(annotation_file, "w") as output_file:
            output_file.write(f"0 {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}\n")
        
        # Visualisation : dessiner la boîte englobante sur l'image
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)  # Rectangle vert pour la boîte
        cv2.putText(image, "Bounding Box", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Afficher l'image avec la boîte
        image = cv2.resize(image, (460,460))
        cv2.imshow(f"Image - {filename}", image)
        
        # Attendre la touche pour fermer la fenêtre
        cv2.waitKey(0)
        cv2.destroyAllWindows()

print("[INFO] Conversion terminée.")
