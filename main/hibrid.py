import cv2
import numpy as np
from ultralytics import YOLO

class HierarchicalDetectionPipeline:
    def __init__(self, yolo_model_path):
        #Inicializa o pipeline.
        #Args:yolo_model_path: Caminho para o modelo YOLO11n treinado (que sabe classificar 'mangueira' e 'bola').
        # 1. Carrega o modelo YOLO para classificação
        self.classifier_model = YOLO(yolo_model_path)
        self.class_names = self.classifier_model.names  # Ex: {0: 'mangueira', 1: 'bola'}

        # 2. Define parâmetros para o primeiro estágio (Segmentação HSV)
        # LIMIARES PARA OBJETOS VERMELHOS/ALARANJADOS (AJUSTE FINO CRÍTICO AQUI)
        # Estes intervalos devem capturar tanto a mangueira vermelha quanto a bola laranja.
        self.lower_red1 = np.array([0, 70, 50])     # Vermelho escuro/Laranja
        self.upper_red1 = np.array([20, 255, 255])  # Ajuste o Hue máximo para pegar laranja
        self.lower_red2 = np.array([160, 70, 50])   # Vermelho vivo
        self.upper_red2 = np.array([180, 255, 255])

        # Kernel para operações morfológicas
        self.kernel = np.ones((5, 5), np.uint8)

        # Resolução para o estágio rápido (quanto menor, mais rápido)
        self.low_res_width = 320

    def _stage1_fast_segmentation(self, frame):

        #Estágio 1: Segmentação rápida em baixa resolução para encontrar candidatos.
        #Returns:list: Lista de bounding boxes [x1, y1, x2, y2] em coordenadas do frame ORIGINAL.
        # 1. Reduz a resolução para processamento rápido
        h, w = frame.shape[:2]
        scale = self.low_res_width / w
        new_w = self.low_res_width
        new_h = int(h * scale)
        small_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # 2. Converte para HSV e aplica limiares
        hsv = cv2.cvtColor(small_frame, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, self.lower_red1, self.upper_red1)
        mask2 = cv2.inRange(hsv, self.lower_red2, self.upper_red2)
        color_mask = cv2.bitwise_or(mask1, mask2)

        # 3. Limpeza morfológica
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, self.kernel, iterations=1)
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, self.kernel, iterations=2)

        # 4. Encontra contornos dos candidatos
        contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidate_boxes = []

        min_area = (new_w * new_h) * 0.001  # Filtra ruídos muito pequenos (0.1% da imagem)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > min_area:
                x, y, w_box, h_box = cv2.boundingRect(cnt)
                # Converte coordenadas de volta para a resolução original
                x1 = int(x / scale)
                y1 = int(y / scale)
                x2 = int((x + w_box) / scale)
                y2 = int((y + h_box) / scale)
                candidate_boxes.append([x1, y1, x2, y2])

        return candidate_boxes

    def _stage2_classify_and_refine(self, original_frame, candidate_boxes):

        #Estágio 2: Classifica cada ROI e refina a detecção.
        final_results = []  # Lista para armazenar resultados

        for box in candidate_boxes:
            x1, y1, x2, y2 = box
            # 1. Recorta a ROI da imagem original para máxima precisão
            roi = original_frame[y1:y2, x1:x2]

            if roi.size == 0:
                continue  # ROI inválida, pula

            # 2. Classifica a ROI usando YOLO
            yolo_results = self.classifier_model(roi, verbose=False, imgsz=320)  # Pode ajustar imgsz
            # Assumimos que o modelo retorna a classe mais confiável na ROI
            if len(yolo_results[0].boxes) > 0:
                boxes_data = yolo_results[0].boxes
                # Pega a detecção com maior confiança nesta ROI
                max_conf_idx = np.argmax(boxes_data.conf.cpu().numpy())
                class_id = int(boxes_data.cls[max_conf_idx])
                confidence = boxes_data.conf[max_conf_idx]
                yolo_box = boxes_data.xyxy[max_conf_idx].cpu().numpy()  # BBox dentro da ROI

                class_name = self.class_names[class_id]
                # Aplica um limiar de confiança
                if confidence > 0.5:  # AJUSTE ESSE VALOR
                    # 3. Refinamento baseado na classe
                    refined_info = None

                    if class_name == 'mangueira':
                        # Refinamento por cor para mangueira (precisão de contorno)
                        refined_contour, refined_center = self._refine_mangueira_by_color(roi)
                        if refined_center:
                            # Converte coordenadas do centro refinado para o frame original
                            center_global = (int(x1 + refined_center[0]), int(y1 + refined_center[1]))
                            refined_info = {'type': 'mangueira', 'center': center_global, 'contour': refined_contour}

                    elif class_name == 'bola':
                        # Para bola, podemos usar a bbox do YOLO ou refinamento por cor laranja
                        # Exemplo: usar o centro da bbox do YOLO
                        yolo_center_x = int((yolo_box[0] + yolo_box[2]) / 2)
                        yolo_center_y = int((yolo_box[1] + yolo_box[3]) / 2)
                        center_global = (x1 + yolo_center_x, y1 + yolo_center_y)
                        refined_info = {'type': 'bola', 'center': center_global, 'bbox': yolo_box}

                    if refined_info:
                        final_results.append(refined_info)

        return final_results

    def _refine_mangueira_by_color(self, roi):
        """
        Refina a detecção da mangueira usando segmentação por cor dentro da ROI.
        Usa limiares mais restritos para vermelho puro.
        """
        # AJUSTE: Use limiares HSV mais específicos para a mangueira vermelha
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lower_mangueira = np.array([0, 120, 70])   # Vermelho mais saturado e escuro
        upper_mangueira = np.array([10, 255, 255])
        lower_mangueira2 = np.array([170, 120, 70])
        upper_mangueira2 = np.array([180, 255, 255])

        mask1 = cv2.inRange(hsv_roi, lower_mangueira, upper_mangueira)
        mask2 = cv2.inRange(hsv_roi, lower_mangueira2, upper_mangueira2)
        mask = cv2.bitwise_or(mask1, mask2)

        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            # Calcula o centro do contorno
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                return largest_contour, (cX, cY)
        return None, None

    def process_frame(self, original_frame):
        """
        Executa o pipeline completo em um frame.
        Returns:
            tuple: (frame_anotado, lista_de_resultados)
        """
        frame_with_results = original_frame.copy()
        results = []

        # --- ESTÁGIO 1: Segmentação Rápida 
        candidate_boxes = self._stage1_fast_segmentation(original_frame)

        # --- ESTÁGIO 2: Classificação e Refino
        if candidate_boxes:  # Só chama o YOLO se houver candidatos
            results = self._stage2_classify_and_refine(original_frame, candidate_boxes)

        # --- ANOTAÇÕES NO FRAME (para visualização)
        for res in results:
            if res['type'] == 'mangueira':
                center = res['center']
                cv2.circle(frame_with_results, center, 10, (0, 0, 255), -1)  # Centro vermelho
                cv2.putText(frame_with_results, 'Mangueira', (center[0]+15, center[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            elif res['type'] == 'bola':
                center = res['center']
                cv2.circle(frame_with_results, center, 8, (0, 255, 255), -1)  # Centro amarelo
                cv2.putText(frame_with_results, 'Bola', (center[0]+15, center[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        return frame_with_results, results

if __name__ == "__main__":
    # 1. Inicializa o pipeline com seu modelo
    pipeline = HierarchicalDetectionPipeline(yolo_model_path='best.pt')

    # 2. Captura de vídeo (webcam ou arquivo)
    cap = cv2.VideoCapture('midia/mang1.mp4')

    print("Pipeline Hierárquico ativo. Pressione 'q' para sair.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 3. Processa o frame completo com o pipeline
        output_frame, detected_objects = pipeline.process_frame(frame)

        # 4. Exibe resultados
        cv2.imshow('Pipeline Hierarquico - Mangueira e Bola', output_frame)

        # Opcional: Log no console
        if detected_objects:
            print(f"Objetos detectados: {len(detected_objects)}")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()