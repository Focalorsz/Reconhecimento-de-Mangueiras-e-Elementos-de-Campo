import cv2
import numpy as np
from ultralytics import YOLO

# 1. Carregar o modelo de SEGMENTAÇÃO
modelo = YOLO('best.pt')

# 2. Ler imagem
imagem = cv2.imread('midia/m1.jpg')
imagem_out = imagem.copy()

# 3. Inferência
resultados = modelo(imagem)

# 4. Verificar se há detecções
if resultados[0].boxes is not None and resultados[0].masks is not None:

    boxes = resultados[0].boxes
    masks = resultados[0].masks

    for i, box in enumerate(boxes):
        confianca = box.conf[0].item()
        classe_id = int(box.cls[0].item())
        nome_classe = modelo.names[classe_id]

        if confianca < 0.7:
            continue

        # --- Bounding box ---
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cv2.rectangle(imagem_out, (x1, y1), (x2, y2), (0, 255, 0), 2)

        label = f'{nome_classe} {confianca:.2f}'
        cv2.putText(
            imagem_out, label, (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
        )

        # Máscara de segmentação
        mask = masks.data[i].cpu().numpy()  # (H, W) binária
        mask = (mask * 255).astype(np.uint8)

        # Criar overlay colorido
        overlay = np.zeros_like(imagem_out)
        overlay[:, :, 1] = mask  # canal verde

        # Sobrepor com transparência
        imagem_out = cv2.addWeighted(imagem_out, 1.0, overlay, 0.5, 0)

# 5. Mostrar resultado
cv2.imshow('YOLO Segmentacao - Linha/Mangueira', imagem_out)
cv2.waitKey(0)
cv2.destroyAllWindows()
