import cv2
import numpy as np
from ultralytics import YOLO

class HierarchicalDetectionPipeline:
    def __init__(self, yolo_model_path):
        # Modelo YOLO
        self.classifier_model = YOLO(yolo_model_path)
        self.class_names = self.classifier_model.names

        # Parâmetros HSV (estágio 1)
        self.lower_red1 = np.array([0, 70, 50])
        self.upper_red1 = np.array([20, 255, 255])
        self.lower_red2 = np.array([160, 70, 50])
        self.upper_red2 = np.array([180, 255, 255])
        self.kernel = np.ones((5, 5), np.uint8)
        self.low_res_width = 320

        # Parâmetros do rastreador (fluxo óptico)
        self.tracks = []               # lista de dicionários: {'id', 'points', 'age', 'type', 'center'}
        self.next_id = 0
        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )
        self.max_age = 5                # número de frames sem detecção para remover track

        # Estado da associação bola-mangueira
        self.target_mangueira_id = None   # ID da mangueira que contém a bola

    def _stage1_fast_segmentation(self, frame):
        h, w = frame.shape[:2]
        scale = self.low_res_width / w
        new_w = self.low_res_width
        new_h = int(h * scale)
        small_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

        hsv = cv2.cvtColor(small_frame, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, self.lower_red1, self.upper_red1)
        mask2 = cv2.inRange(hsv, self.lower_red2, self.upper_red2)
        color_mask = cv2.bitwise_or(mask1, mask2)

        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, self.kernel, iterations=1)
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, self.kernel, iterations=2)

        contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidate_boxes = []
        min_area = (new_w * new_h) * 0.001

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > min_area:
                x, y, w_box, h_box = cv2.boundingRect(cnt)
                x1 = int(x / scale)
                y1 = int(y / scale)
                x2 = int((x + w_box) / scale)
                y2 = int((y + h_box) / scale)
                candidate_boxes.append([x1, y1, x2, y2])

        return candidate_boxes

    def _stage2_classify_and_refine(self, original_frame, candidate_boxes):
        final_results = []

        for box in candidate_boxes:
            x1, y1, x2, y2 = box
            roi = original_frame[y1:y2, x1:x2]
            if roi.size == 0:
                continue

            yolo_results = self.classifier_model(roi, verbose=False, imgsz=320)
            if len(yolo_results[0].boxes) == 0:
                continue

            boxes_data = yolo_results[0].boxes
            max_conf_idx = np.argmax(boxes_data.conf.cpu().numpy())
            class_id = int(boxes_data.cls[max_conf_idx])
            confidence = boxes_data.conf[max_conf_idx]
            class_name = self.class_names[class_id]

            if confidence > 0.5:
                refined_info = None
                if class_name == 'mangueira':
                    refined_contour, refined_center = self._refine_mangueira_by_color(roi)
                    if refined_center:
                        center_global = (int(x1 + refined_center[0]), int(y1 + refined_center[1]))
                        refined_info = {
                            'type': 'mangueira',
                            'center': center_global,
                            'contour': refined_contour,
                            'confidence': confidence
                        }
                elif class_name == 'bola':
                    # centro da bounding box do YOLO
                    yolo_box = boxes_data.xyxy[max_conf_idx].cpu().numpy()
                    yolo_center_x = int((yolo_box[0] + yolo_box[2]) / 2)
                    yolo_center_y = int((yolo_box[1] + yolo_box[3]) / 2)
                    center_global = (x1 + yolo_center_x, y1 + yolo_center_y)
                    refined_info = {
                        'type': 'bola',
                        'center': center_global,
                        'confidence': confidence
                    }

                if refined_info:
                    final_results.append(refined_info)

        return final_results

    def _refine_mangueira_by_color(self, roi):
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lower = np.array([0, 120, 70])
        upper = np.array([10, 255, 255])
        lower2 = np.array([170, 120, 70])
        upper2 = np.array([180, 255, 255])
        mask1 = cv2.inRange(hsv_roi, lower, upper)
        mask2 = cv2.inRange(hsv_roi, lower2, upper2)
        mask = cv2.bitwise_or(mask1, mask2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                return largest, (cX, cY)
        return None, None

    def _track_objects(self, prev_gray, curr_gray, detections):
        """
        Atualiza os tracks com base nas novas detecções e fluxo óptico.
        detections: lista de dict com 'center' e 'type' (vindo do estágio 2).
        Retorna lista de tracks atualizada.
        """
        if not self.tracks:
            # Primeiro frame: cria tracks para todas as detecções
            for det in detections:
                self.tracks.append({
                    'id': self.next_id,
                    'points': np.array([det['center']], dtype=np.float32).reshape(-1,1,2),
                    'age': 0,
                    'type': det['type'],
                    'center': det['center']
                })
                self.next_id += 1
            return self.tracks

        # 1. Projetar pontos dos tracks atuais usando fluxo óptico
        prev_points = np.array([t['points'][-1] for t in self.tracks], dtype=np.float32).reshape(-1,1,2)
        curr_points, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_points, None, **self.lk_params)

        # 2. Atualizar tracks com os pontos projetados (se bem sucedidos)
        new_tracks = []
        for i, track in enumerate(self.tracks):
            if status[i][0] == 1:
                new_point = curr_points[i].reshape(-1,2)[0]
                track['points'] = np.vstack([track['points'], new_point.reshape(-1,2)]) if track['points'] is not None else new_point.reshape(-1,2)
                track['center'] = tuple(new_point.astype(int))
                track['age'] += 1
                new_tracks.append(track)
            else:
                # Ponto perdido: não mantém
                pass

        # 3. Associar novas detecções aos tracks existentes
        unmatched_detections = []
        for det in detections:
            best_match = None
            best_dist = float('inf')
            for track in new_tracks:
                if track['type'] == det['type']:  # só associa objetos do mesmo tipo
                    dist = np.linalg.norm(np.array(track['center']) - np.array(det['center']))
                    if dist < best_dist and dist < 50:  # limiar de distância (pixels)
                        best_dist = dist
                        best_match = track
            if best_match:
                # Atualiza track com a nova detecção (substitui o ponto)
                best_match['points'] = np.array([det['center']], dtype=np.float32).reshape(-1,2)
                best_match['center'] = det['center']
                best_match['age'] = 0
            else:
                unmatched_detections.append(det)

        # 4. Criar novos tracks para detecções não associadas
        for det in unmatched_detections:
            new_tracks.append({
                'id': self.next_id,
                'points': np.array([det['center']], dtype=np.float32).reshape(-1,2),
                'age': 0,
                'type': det['type'],
                'center': det['center']
            })
            self.next_id += 1

        # 5. Remover tracks muito antigos (age > max_age)
        self.tracks = [t for t in new_tracks if t['age'] <= self.max_age]
        return self.tracks

    def _associate_bola_to_mangueira(self):
        """
        Define qual mangueira está com a bola (target_mangueira_id).
        Baseia-se na distância entre centros de bolas e mangueiras nos tracks atuais.
        """
        bolas = [t for t in self.tracks if t['type'] == 'bola']
        mangueiras = [t for t in self.tracks if t['type'] == 'mangueira']

        if not bolas or not mangueiras:
            # Se não há bola ou mangueira, mantém o target anterior, mas pode ser None
            return

        # Para simplificar, pega a primeira bola (se houver múltiplas, considera a mais próxima da mangueira alvo)
        # Ideal: para cada bola, encontrar a mangueira mais próxima e atualizar o target
        best_pair = None
        min_dist = float('inf')
        for bola in bolas:
            for mang in mangueiras:
                dist = np.linalg.norm(np.array(bola['center']) - np.array(mang['center']))
                if dist < min_dist:
                    min_dist = dist
                    best_pair = (bola, mang)

        if best_pair and min_dist < 150:  # limiar de distância para considerar associação
            self.target_mangueira_id = best_pair[1]['id']
        else:
            self.target_mangueira_id = None

    def process_frame(self, original_frame, prev_gray=None):
        """
        Processa um frame e retorna o frame anotado, resultados e (opcional) o gray atual.
        Se prev_gray for None, assume que é o primeiro frame e não faz tracking.
        """
        frame_with_results = original_frame.copy()
        gray = cv2.cvtColor(original_frame, cv2.COLOR_BGR2GRAY)

        # Estágio 1
        candidate_boxes = self._stage1_fast_segmentation(original_frame)

        # Estágio 2
        detections = []
        if candidate_boxes:
            detections = self._stage2_classify_and_refine(original_frame, candidate_boxes)

        # Rastreamento
        if prev_gray is not None:
            self._track_objects(prev_gray, gray, detections)
        else:
            # Primeiro frame: inicializa tracks com as detecções
            for det in detections:
                self.tracks.append({
                    'id': self.next_id,
                    'points': np.array([det['center']], dtype=np.float32).reshape(-1,2),
                    'age': 0,
                    'type': det['type'],
                    'center': det['center']
                })
                self.next_id += 1

        # Associação bola-mangueira
        self._associate_bola_to_mangueira()

        # Anotação dos tracks
        for track in self.tracks:
            center = track['center']
            if track['type'] == 'mangueira':
                # cor padrão vermelho, mas se for a alvo, muda para azul
                color = (0, 255, 0) if track['id'] == self.target_mangueira_id else (0, 0, 255)
                cv2.circle(frame_with_results, center, 10, color, -1)
                label = f"Mangueira {track['id']}"
                cv2.putText(frame_with_results, label, (center[0]+15, center[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            elif track['type'] == 'bola':
                cv2.circle(frame_with_results, center, 8, (0, 255, 255), -1)
                cv2.putText(frame_with_results, f"Bola {track['id']}", (center[0]+15, center[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Opcional: desenhar linha do track (histórico)
        for track in self.tracks:
            if len(track['points']) > 1:
                pts = track['points'].astype(np.int32).reshape(-1,1,2)
                cv2.polylines(frame_with_results, [pts], False, (255,255,255), 1)

        return frame_with_results, self.tracks, gray

if __name__ == "__main__":
    pipeline = HierarchicalDetectionPipeline(yolo_model_path='best.pt')
    cap = cv2.VideoCapture('midia/mang1.mp4')

    prev_gray = None
    print("Pipeline com rastreamento ativo. Pressione 'q' para sair.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        output_frame, tracks, gray = pipeline.process_frame(frame, prev_gray)
        prev_gray = gray

        cv2.imshow('Pipeline com Rastreamento', output_frame)
        if tracks:
            print(f"Tracks ativos: {len(tracks)} | Alvo: {pipeline.target_mangueira_id}")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()