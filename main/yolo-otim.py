import cv2
import numpy as np
from ultralytics import YOLO

# CONFIGURAÇÕES
YOLO_INTERVAL = 10      # Executa YOLO a cada N frames
CONF_MIN = 0.7

# MODELO
modelo = YOLO('best.pt')

# VARIÁVEIS DE ESTADO
frame_count = 0
ultimo_resultado = None  # Guarda última saída do YOLO

# FUNÇÃO DE PROCESSAMENTO
def processar_frame_yolo(frame):
    #Executa YOLO-seg e extrai informações da mangueira
    
    resultados = modelo(frame, verbose=False)

    if resultados[0].masks is None:
        return None

    boxes = resultados[0].boxes
    masks = resultados[0].masks

    deteccoes = []

    for i, box in enumerate(boxes):
        conf = box.conf[0].item()
        if conf < CONF_MIN:
            continue

        class_id = int(box.cls[0])
        class_name = modelo.names[class_id]

        mask = masks.data[i].cpu().numpy()
        mask_bin = (mask > 0.5).astype(np.uint8) * 255

        contours, _ = cv2.findContours(
            mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            continue

        contorno = max(contours, key=cv2.contourArea)

        M = cv2.moments(contorno)
        if M["m00"] == 0:
            continue

        centro = (
            int(M["m10"] / M["m00"]),
            int(M["m01"] / M["m00"])
        )

        deteccoes.append({
            'classe': class_name,
            'confianca': conf,
            'contorno': contorno,
            'centro': centro
        })

    return deteccoes


# LOOP DE VÍDEO
cap = cv2.VideoCapture('midia/mang1.mp4')  # ou 0 para webcam

print("YOLO-seg rodando a cada", YOLO_INTERVAL, "frames")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_out = frame.copy()

    # Executa YOLO apenas a cada N frames
    if frame_count % YOLO_INTERVAL == 0:
        deteccoes = processar_frame_yolo(frame)
        if deteccoes:
            ultimo_resultado = deteccoes

    # Desenha último resultado conhecido
    if ultimo_resultado:
        for det in ultimo_resultado:
            centro = det['centro']
            contorno = det['contorno']

            cv2.drawContours(frame_out, [contorno], -1, (0, 255, 0), 2)
            cv2.circle(frame_out, centro, 6, (0, 0, 255), -1)

            cv2.putText(
                frame_out, det['classe'],
                (centro[0] + 10, centro[1]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
            )

    cv2.imshow('YOLO-seg Otimizado', frame_out)

    frame_count += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
