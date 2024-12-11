import cv2
import os
import random
import shutil

def annotate_image(image_path):
    """
    Fonction pour annoter une image.
    L'utilisateur clique et trace des rectangles autour des zones d'intérêt.
    """
    global roi_coords
    roi_coords = []  # Liste pour stocker toutes les annotations

    def draw_rectangle(event, x, y, flags, param):
        global roi_coords, temp_image
        if event == cv2.EVENT_LBUTTONDOWN:
            roi_coords.append([(x, y)])  # Début d'un rectangle
        elif event == cv2.EVENT_LBUTTONUP:
            roi_coords[-1].append((x, y))  # Fin du rectangle
            # Dessiner le rectangle
            cv2.rectangle(temp_image, roi_coords[-1][0], roi_coords[-1][1], (255, 0, 0), 2)
            cv2.imshow("Image", temp_image)

    # Charger l'image
    image = cv2.imread(image_path)
    image = cv2.resize(image, (640, 640))
    global temp_image
    temp_image = image.copy()
    cv2.imshow("Image", image)

    # Associer la fonction de callback
    cv2.setMouseCallback("Image", draw_rectangle)

    # Attendre l'entrée utilisateur
    print("[INFO] Tracez des rectangles autour des zones d'intérêt.")
    print("[INFO] Appuyez sur 'c' pour confirmer ou 'q' pour quitter.")
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord("c"):  # Confirmer
            break
        elif key == ord("q"):  # Quitter
            roi_coords = []
            break

    cv2.destroyAllWindows()
    return roi_coords


# Répertoires des images et annotations
source_dir = "datasets/dataset/train_segment"
image_dir = "test_annot/images/"  # Répertoire contenant les images
output_dir = "test_annot/labels/"  # Répertoire pour sauvegarder les annotations
NB_FICHIERS = 50 # Nombre d'images à annoter
if os.path.exists(image_dir):
    shutil.rmtree(image_dir)
os.makedirs(image_dir)
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir)


files = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]
random.shuffle(files)
train_files = files[:NB_FICHIERS]

# Fonction pour déplacer les fichiers
def move_files(file_list, destination_dir):
    for file in file_list:
        shutil.copy(os.path.join(source_dir, file), os.path.join(destination_dir, file))

# Déplacer les fichiers
move_files(train_files, image_dir)

compteur = 0

# Parcourir les images
for image_file in os.listdir(image_dir):
    image_path = os.path.join(image_dir, image_file)
    output_path = os.path.join(output_dir, image_file.replace(".jpg", ".txt"))

    # Vérifier si une annotation existe déjà
    if os.path.exists(output_path):
        print(f"[INFO] Annotation existante pour {image_file}, sautée.")
        continue

    # Annoter l'image
    rois = annotate_image(image_path)
    image = cv2.imread(image_path)
    if rois:
        with open(output_path, "w") as f:
            for roi in rois:

                x_min = min(roi[0][0], roi[1][0])
                y_min = min(roi[0][1], roi[1][1])
                x_max = max(roi[0][0], roi[1][0])
                y_max = max(roi[0][1], roi[1][1])
            
                # Normaliser les coordonnées pour le format YOLO
                x_center = (x_min + x_max) / 2 / image.shape[1]
                y_center = (y_min + y_max) / 2 / image.shape[0]
                width = (x_max - x_min) / image.shape[1]
                height = (y_max - y_min) / image.shape[0]

                # Écrire les annotations
                f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
        print(f"[INFO] Annotations sauvegardées pour {image_file}.")
        compteur += 1 
        print(" NOMBRE D'IMAGES TRAITÉES : ", compteur)
    else:
        print(f"[INFO] Aucune annotation pour {image_file}.")

print("[INFO] Annotation terminée.")
