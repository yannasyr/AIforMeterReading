import cv2
import os

# Répertoires des images et annotations
image_dir = "datasets/YOLO-training/images200/"  # Répertoire contenant les images
output_dir = "datasets/YOLO-training/labels200/"  # Répertoire pour sauvegarder les annotations

def annotate_image(image_path):
    """
    Fonction pour annoter une image.
    L'utilisateur clique et trace un rectangle autour de la zone d'intérêt.
    """
    global roi_coords
    roi_coords = []

    def draw_rectangle(event, x, y, flags, param):
        global roi_coords
        if event == cv2.EVENT_LBUTTONDOWN:
            roi_coords = [(x, y)]  # Coordonnées de départ
        elif event == cv2.EVENT_LBUTTONUP:
            roi_coords.append((x, y))  # Coordonnées de fin
            # Dessiner le rectangle
            cv2.rectangle(temp_image, roi_coords[0], roi_coords[1], (255, 0, 0), 2)
            cv2.imshow("Image", temp_image)

    # Charger l'image
    image = cv2.imread(image_path)
    image = cv2.resize(image, (640,640))
    temp_image = image.copy()
    cv2.imshow("Image", image)

    # Associer la fonction de callback
    cv2.setMouseCallback("Image", draw_rectangle)

    # Attendre l'entrée utilisateur
    print("[INFO] Tracez un rectangle autour de la zone d'intérêt, puis appuyez sur 'c' pour confirmer ou 'q' pour quitter.")
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord("c"):  # Confirmer
            break
        elif key == ord("q"):  # Quitter
            roi_coords = []
            break

    cv2.destroyAllWindows()
    return roi_coords

# Parcourir les images
for image_file in os.listdir(image_dir):
    image_path = os.path.join(image_dir, image_file)
    output_path = os.path.join(output_dir, image_file.replace(".jpg", ".txt"))

    # Vérifier si une annotation existe déjà
    if os.path.exists(output_path):
        print(f"[INFO] Annotation existante pour {image_file}, sautée.")
        continue

    # Annoter l'image
    roi = annotate_image(image_path)
    image = cv2.imread(image_path)
    image = cv2.resize(image, (640,640))
    if roi:
        # Normaliser les coordonnées pour le format YOLO
        x_min, y_min = roi[0]
        x_max, y_max = roi[1]
        x_center = (x_min + x_max) / 2 / image.shape[1]
        y_center = (y_min + y_max) / 2 / image.shape[0]
        width = (x_max - x_min) / image.shape[1]
        height = (y_max - y_min) / image.shape[0]

        # Sauvegarder les annotations
        with open(output_path, "w") as f:
            f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
        print(f"[INFO] Annotation sauvegardée pour {image_file}.")
    else:
        print(f"[INFO] Aucune annotation pour {image_file}.")

print("[INFO] Annotation terminée.")
