import cv2
import numpy as np

def segmentar_vermelho_com_abertura(imagem_bgr):
    # Segmenta tons de vermelho (claro a escuro) e aplica abertura morfológica.
    
    #Args:imagem_bgr: Imagem de entrada no formato BGR (padrão OpenCV). Talvez seja rgb no collab ou no codespaces.
    
    #Returns:tuple: (máscara_final, resultado_com_contornos)
    
    # Converte BGR para HSV
    hsv = cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2HSV)
    
    # Intervalos para o vermelho (Hue: ~0-10 e ~170-180)
    # Os valores são ajustáveis: [Hue, Saturação, Valor]
    vermelho_inf1 = np.array([0, 70, 50])
    vermelho_sup1 = np.array([10, 255, 255])
    vermelho_inf2 = np.array([170, 70, 50])
    vermelho_sup2 = np.array([180, 255, 255])
    
    # Cria máscaras para cada faixa e combiná elas
    mascara1 = cv2.inRange(hsv, vermelho_inf1, vermelho_sup1)
    mascara2 = cv2.inRange(hsv, vermelho_inf2, vermelho_sup2)
    mascara_vermelha = cv2.bitwise_or(mascara1, mascara2)
    
    # Aplica ABERTURA MORFOLÓGICA (Erosão seguida de Dilatação)
    # Remove pequenos ruídos brancos (pixels isolados)
    kernel = np.ones((5,5), np.uint8) # Kernel estruturante 5x5
    # A função cv.morphologyEx com o parâmetro cv.MORPH_OPEN realiza a abertura[citation:9].
    mascara_limpa = cv2.morphologyEx(mascara_vermelha, cv.MORPH_OPEN, kernel, iterations=2)
    
    # 5. (Opcional, se quiser pode deixar comentado) Encontrar contornos e desenhar na imagem original
    resultado = imagem_bgr.copy()
    contornos, _ = cv2.findContours(mascara_limpa, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for cnt in contornos:
        # Filtrar contornos muito pequenos
        area = cv2.contourArea(cnt)
        if area > 500: # Ajuste o limite mínimo de área conforme necessário
            (x, y), raio = cv2.minEnclosingCircle(cnt)
            centro = (int(x), int(y))
            raio = int(raio)
            # Desenhar círculo envolvente
            cv2.circle(resultado, centro, raio, (0, 255, 0), 2)
            # Desenhar um ponto no centro
            cv2.circle(resultado, centro, 5, (0, 0, 255), -1)
    
    return mascara_limpa, resultado

# Carregar uma imagem
# img = cv2.imread('imagem da mangueira.jpg')
# Para testar com webcam:
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    mascara, resultado = segmentar_vermelho_com_abertura(frame)
    cv2.imshow('Mascara Limpa (Abertura)', mascara)
    cv2.imshow('Resultado com Deteccao', resultado)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()