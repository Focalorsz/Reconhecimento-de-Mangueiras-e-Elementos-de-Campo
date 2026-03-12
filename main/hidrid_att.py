import cv2
import numpy as np
from ultralytics import YOLO
import time


class RaspberryPiOptimizedDetector:
    """Pipeline híbrido: Blob Detection + HSV + YOLO com ROI dinâmico."""
    
    def __init__(self, yolo_model_path, yolo_interval=10):
        """
        Inicializa o detector híbrido otimizado.
        
        Args:
            yolo_model_path (str): Caminho para o modelo YOLO treinado (.pt)
            yolo_interval (int): Executar YOLO a cada N frames (padrão: 10)
        """
        print("[INFO] Inicializando detector...")
        
        self.yolo_model = YOLO(yolo_model_path)
        self.class_names = self.yolo_model.names
        self.yolo_interval = yolo_interval
        
        self.frame_count = 0
        self.tracked_objects = []
        
        self.lower_orange = np.array([5, 100, 100])
        self.upper_orange = np.array([25, 255, 255])
        
        self.lower_red1 = np.array([0, 100, 50])
        self.upper_red1 = np.array([10, 255, 255])
        self.lower_red2 = np.array([170, 100, 50])
        self.upper_red2 = np.array([180, 255, 255])
        
        self.kernel_small = np.ones((3, 3), np.uint8)
        self.kernel_medium = np.ones((5, 5), np.uint8)
        
        self.blob_detector = self._setup_blob_detector()
        self.fps_history = []
        
        print("[INFO] Detector inicializado!")
    
    def _setup_blob_detector(self):
        """
        Configura o SimpleBlobDetector para detectar objetos circulares (bolas).
        
        Returns:
            cv2.SimpleBlobDetector: Detector configurado com filtros de circularidade
        """
        params = cv2.SimpleBlobDetector_Params()
        params.filterByColor = True
        params.blobColor = 255
        params.filterByArea = True
        params.minArea = 100
        params.maxArea = 50000
        params.filterByCircularity = True
        params.minCircularity = 0.7
        params.filterByConvexity = True
        params.minConvexity = 0.8
        params.filterByInertia = True
        params.minInertiaRatio = 0.6
        return cv2.SimpleBlobDetector_create(params)
    
    def _detect_ball_blob(self, hsv):
        """
        Detecta bolas laranjas usando segmentação HSV e Blob Detection.
        
        Args:
            hsv (numpy.ndarray): Frame convertido para espaço de cores HSV
            
        Returns:
            list: Lista de tuplas (x, y, radius) com as posições e raios das bolas detectadas
        """
        mask_orange = cv2.inRange(hsv, self.lower_orange, self.upper_orange)
        mask_orange = cv2.morphologyEx(mask_orange, cv2.MORPH_OPEN, self.kernel_small, iterations=2)
        mask_orange = cv2.morphologyEx(mask_orange, cv2.MORPH_CLOSE, self.kernel_medium, iterations=2)
        
        keypoints = self.blob_detector.detect(mask_orange)
        
        detections = []
        for kp in keypoints:
            x, y = int(kp.pt[0]), int(kp.pt[1])
            radius = int(kp.size / 2)
            detections.append((x, y, radius))
        
        return detections
    
    def _detect_hose_hsv(self, hsv):
        """
        Detecta mangueiras vermelhas usando segmentação HSV em dois ranges.
        
        Args:
            hsv (numpy.ndarray): Frame convertido para espaço de cores HSV
            
        Returns:
            tuple: (mask, bbox, center) onde:
                - mask: Máscara binária da detecção
                - bbox: Bounding box (x, y, w, h)
                - center: Centro (cx, cy) ou None se não detectado
        """
        mask1 = cv2.inRange(hsv, self.lower_red1, self.upper_red1)
        mask2 = cv2.inRange(hsv, self.lower_red2, self.upper_red2)
        mask_red = cv2.bitwise_or(mask1, mask2)
        
        mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, self.kernel_small, iterations=2)
        mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, self.kernel_medium, iterations=3)
        
        contours, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None, None, None
        
        largest_contour = max(contours, key=cv2.contourArea)
        
        if cv2.contourArea(largest_contour) < 500:
            return None, None, None
        
        x, y, w, h = cv2.boundingRect(largest_contour)
        bbox = (x, y, w, h)
        
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            center = (cx, cy)
        else:
            center = (x + w//2, y + h//2)
        
        return mask_red, bbox, center
    
    def _get_roi_from_masks(self, frame, mask_red, ball_detections):
        """
        Calcula a região de interesse (ROI) dinâmica baseada nas detecções HSV/Blob.
        Engloba todas as detecções com margem adicional de 20%.
        
        Args:
            frame (numpy.ndarray): Frame original
            mask_red (numpy.ndarray): Máscara da mangueira detectada (ou None)
            ball_detections (list): Lista de detecções de bolas
            
        Returns:
            tuple: (roi_bbox, roi_frame) onde:
                - roi_bbox: Coordenadas (x, y, w, h) do ROI
                - roi_frame: Recorte do frame na região do ROI
                Retorna (None, None) se não houver detecções válidas
        """
        h, w = frame.shape[:2]
        x_min, y_min = w, h
        x_max, y_max = 0, 0
        
        if mask_red is not None:
            coords = cv2.findNonZero(mask_red)
            if coords is not None:
                x, y, w_mask, h_mask = cv2.boundingRect(coords)
                x_min = min(x_min, x)
                y_min = min(y_min, y)
                x_max = max(x_max, x + w_mask)
                y_max = max(y_max, y + h_mask)
        
        for (bx, by, radius) in ball_detections:
            margin = radius * 2
            x_min = min(x_min, max(0, bx - margin))
            y_min = min(y_min, max(0, by - margin))
            x_max = max(x_max, min(w, bx + margin))
            y_max = max(y_max, min(h, by + margin))
        
        if x_max <= x_min or y_max <= y_min:
            return None, None
        
        margin_x = int((x_max - x_min) * 0.2)
        margin_y = int((y_max - y_min) * 0.2)
        
        x_min = max(0, x_min - margin_x)
        y_min = max(0, y_min - margin_y)
        x_max = min(w, x_max + margin_x)
        y_max = min(h, y_max + margin_y)
        
        roi_bbox = (x_min, y_min, x_max - x_min, y_max - y_min)
        roi_frame = frame[y_min:y_max, x_min:x_max]
        
        return roi_bbox, roi_frame
    
    def _run_yolo_on_roi(self, roi_frame, roi_bbox):
        """
        Executa o modelo YOLO apenas na região de interesse (ROI).
        Converte as coordenadas das detecções de volta para o frame completo.
        
        Args:
            roi_frame (numpy.ndarray): Recorte do frame (ROI)
            roi_bbox (tuple): Coordenadas do ROI no frame original (x, y, w, h)
            
        Returns:
            list: Lista de dicionários com detecções contendo:
                - bbox: (x1, y1, x2, y2) em coordenadas globais
                - class: Nome da classe detectada
                - confidence: Confiança da detecção
                - center: Centro (x, y) do objeto
        """
        if roi_frame.size == 0:
            return []
        
        x_offset, y_offset = roi_bbox[0], roi_bbox[1]
        results = self.yolo_model(roi_frame, verbose=False, conf=0.5, imgsz=320)
        
        detections = []
        if len(results[0].boxes) > 0:
            boxes = results[0].boxes
            xyxy = boxes.xyxy.cpu().numpy().astype(int)
            cls = boxes.cls.cpu().numpy().astype(int)
            conf = boxes.conf.cpu().numpy()
            
            for i in range(len(boxes)):
                x1, y1, x2, y2 = xyxy[i]
                x1_global = x1 + x_offset
                y1_global = y1 + y_offset
                x2_global = x2 + x_offset
                y2_global = y2 + y_offset
                
                class_id = cls[i]
                confidence = float(conf[i])
                class_name = self.class_names[class_id]
                
                detections.append({
                    'bbox': (x1_global, y1_global, x2_global, y2_global),
                    'class': class_name,
                    'confidence': confidence,
                    'center': ((x1_global + x2_global) // 2, (y1_global + y2_global) // 2)
                })
        
        return detections
    
    def _update_tracking(self, current_detections):
        """
        Atualiza a lista de objetos rastreados com as detecções mais recentes do YOLO.
        
        Args:
            current_detections (list): Lista de detecções do YOLO no frame atual
        """
        if not current_detections:
            return
        self.tracked_objects = current_detections
    
    def _interpolate_tracking(self):
        """
        Retorna os objetos rastreados para uso nos frames intermediários (sem YOLO).
        
        Returns:
            list: Lista de objetos rastreados da última execução do YOLO
        """
        return self.tracked_objects
    
    def process_frame(self, frame):
        """
        Processa um frame completo através do pipeline híbrido:
        1. Detecção rápida com HSV/Blob
        2. Criação de ROI dinâmico
        3. YOLO condicional no ROI (a cada N frames)
        4. Desenho de todas as detecções e informações
        
        Args:
            frame (numpy.ndarray): Frame BGR do vídeo
            
        Returns:
            tuple: (frame_output, results, avg_fps) onde:
                - frame_output: Frame anotado com todas as detecções
                - results: Dicionário com 'balls', 'hose', 'yolo'
                - avg_fps: FPS médio calculado
        """
        start_time = time.time()
        frame_output = frame.copy()
        h, w = frame.shape[:2]
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        ball_detections = self._detect_ball_blob(hsv)
        mask_red, hose_bbox, hose_center = self._detect_hose_hsv(hsv)
        
        should_run_yolo = (self.frame_count % self.yolo_interval == 0)
        
        if should_run_yolo and (mask_red is not None or ball_detections):
            roi_bbox, roi_frame = self._get_roi_from_masks(frame, mask_red, ball_detections)
            
            if roi_bbox is not None and roi_frame is not None:
                yolo_detections = self._run_yolo_on_roi(roi_frame, roi_bbox)
                self._update_tracking(yolo_detections)
                
                x, y, w_roi, h_roi = roi_bbox
                cv2.rectangle(frame_output, (x, y), (x + w_roi, y + h_roi), (0, 0, 255), 1)
                cv2.putText(frame_output, 'ROI', (x, y - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        
        # Desenha bolas (Blob)
        for (bx, by, radius) in ball_detections:
            cv2.circle(frame_output, (bx, by), radius, (0, 255, 255), 2)
            cv2.circle(frame_output, (bx, by), 3, (0, 0, 255), -1)
            cv2.putText(frame_output, 'Bola (Blob)', (bx + radius + 5, by),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        # Desenha mangueira (HSV) - Contorno azul
        if hose_bbox is not None:
            x, y, w_hose, h_hose = hose_bbox
            cv2.rectangle(frame_output, (x, y), (x + w_hose, y + h_hose), (255, 0, 0), 2)
            cv2.circle(frame_output, hose_center, 5, (255, 0, 0), -1)
            cv2.putText(frame_output, 'Mangueira (HSV)', (x, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # Desenha detecções YOLO rastreadas - Caixas verdes
        tracked = self._interpolate_tracking()
        for det in tracked:
            x1, y1, x2, y2 = det['bbox']
            cv2.rectangle(frame_output, (x1, y1), (x2, y2), (0, 255, 0), 3)
            
            label = f"{det['class']} {det['confidence']:.2f} (YOLO)"
            cv2.putText(frame_output, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        elapsed_time = time.time() - start_time
        fps = 1.0 / elapsed_time if elapsed_time > 0 else 0
        self.fps_history.append(fps)
        
        if len(self.fps_history) > 30:
            self.fps_history.pop(0)
        
        avg_fps = np.mean(self.fps_history)
        
        # Info overlay
        info_y = 30
        cv2.putText(frame_output, f'FPS: {avg_fps:.1f}', (10, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        yolo_status = "YOLO: ON" if should_run_yolo else f"YOLO: OFF ({self.frame_count % self.yolo_interval}/{self.yolo_interval})"
        cv2.putText(frame_output, yolo_status, (10, info_y + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        cv2.putText(frame_output, f'Frame: {self.frame_count}', (10, info_y + 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        self.frame_count += 1
        
        results = {
            'balls': ball_detections,
            'hose': (hose_bbox, hose_center) if hose_bbox else None,
            'yolo': tracked
        }
        
        return frame_output, results, avg_fps


def main():
    print("=" * 60)
    print("RASPBERRY PI 5 OPTIMIZED DETECTOR")
    print("=" * 60)
    
    MODEL_PATH = 'main/best .pt'
    VIDEO_PATH = 'main/midia/mang1.mp4'
    YOLO_INTERVAL = 10
    
    # Inicializa detector
    detector = RaspberryPiOptimizedDetector(
        yolo_model_path=MODEL_PATH,
        yolo_interval=YOLO_INTERVAL
    )
    
    cap = cv2.VideoCapture(VIDEO_PATH)
    
    if not cap.isOpened():
        print(f"[ERRO] Não foi possível abrir: {VIDEO_PATH}")
        return
    
    fps_original = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"\n[INFO] Vídeo: {VIDEO_PATH}")
    print(f"[INFO] Resolução: {width}x{height}")
    print(f"[INFO] FPS Original: {fps_original:.2f}")
    print(f"[INFO] Total de Frames: {total_frames}")
    print(f"[INFO] YOLO executará a cada {YOLO_INTERVAL} frames")
    print("\n" + "=" * 60)
    print("CONTROLES:")
    print("  'q' → Sair")
    print("  'p' → Pausar/Retomar")
    print("  '+' → Aumentar velocidade (2x)")
    print("  '-' → Diminuir velocidade (0.5x)")
    print("  'r' → Resetar velocidade (1x)")
    print("=" * 60 + "\n")
    
    base_delay = int(1000 / fps_original) if fps_original > 0 else 30
    delay = base_delay
    speed_multiplier = 1.0
    
    paused = False
    
    while True:
        if not paused:
            ret, frame = cap.read()
            
            if not ret:
                print("\n[INFO] Fim do vídeo.")
                break
            
            output_frame, detections, fps = detector.process_frame(frame)
            cv2.imshow('Raspberry Pi 5 - Optimized Detection', output_frame)
        key = cv2.waitKey(delay) & 0xFF
        
        if key == ord('q'):
            print("\n[INFO] Encerrando...")
            break
        elif key == ord('p'):
            paused = not paused
            status = "PAUSADO" if paused else "RETOMADO"
            print(f"[INFO] Vídeo {status}")
        elif key == ord('+') or key == ord('='):
            speed_multiplier *= 2.0
            delay = max(1, int(base_delay / speed_multiplier))
            print(f"[INFO] Velocidade: {speed_multiplier:.1f}x")
        elif key == ord('-') or key == ord('_'):
            speed_multiplier /= 2.0
            delay = max(1, int(base_delay / speed_multiplier))
            print(f"[INFO] Velocidade: {speed_multiplier:.1f}x")
        elif key == ord('r'):
            speed_multiplier = 1.0
            delay = base_delay
            print(f"[INFO] Velocidade resetada: {speed_multiplier:.1f}x")
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Estatísticas
    if detector.fps_history:
        print("\n" + "=" * 60)
        print("ESTATÍSTICAS DE PERFORMANCE")
        print("=" * 60)
        print(f"FPS Médio: {np.mean(detector.fps_history):.2f}")
        print(f"FPS Mínimo: {np.min(detector.fps_history):.2f}")
        print(f"FPS Máximo: {np.max(detector.fps_history):.2f}")
        print(f"Total de Frames Processados: {detector.frame_count}")
        print("=" * 60)


if __name__ == "__main__":
    main()
    