import cv2
import mediapipe as mp
import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
import io
import base64
from typing import Optional

router = APIRouter()

# Configuración de MediaPipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# Estilos de colores
style_red = mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=4)   # Mano izquierda = rojo
style_blue = mp_draw.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=4)   # Mano derecha = azul

hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7)

def contar_dedos(hand_landmarks, hand_label):
    """Función para contar dedos (solo números exactos del 1 al 5)"""
    lm = hand_landmarks.landmark
    dedos = []

    # Pulgar (depende si es izquierda o derecha)
    if hand_label == "Right":
        dedos.append(1 if lm[4].x < lm[3].x else 0)
    else:
        dedos.append(1 if lm[4].x > lm[3].x else 0)

    # Otros dedos (arriba si punta está más arriba que nudillo)
    dedos.append(1 if lm[8].y < lm[6].y else 0)   # índice
    dedos.append(1 if lm[12].y < lm[10].y else 0) # medio
    dedos.append(1 if lm[16].y < lm[14].y else 0) # anular
    dedos.append(1 if lm[20].y < lm[18].y else 0) # meñique

    total = sum(dedos)

    # Validación estricta
    if total == 1 and dedos == [0,1,0,0,0]: return 1
    if total == 2 and dedos == [0,1,1,0,0]: return 2
    if total == 3 and dedos == [0,1,1,1,0]: return 3
    if total == 4 and dedos == [0,1,1,1,1]: return 4
    if total == 5 and dedos == [1,1,1,1,1]: return 5

    return 0  # Nada válido

def procesar_operacion(frame, operacion):
    """Procesa una operación matemática en el frame"""
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    izq_num, der_num = 0, 0

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            label = handedness.classification[0].label  # "Left" o "Right"

            if label == "Left":
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS, style_red, style_red)
                izq_num = contar_dedos(hand_landmarks, label)
            else:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS, style_blue, style_blue)
                der_num = contar_dedos(hand_landmarks, label)

    # Mostrar números
    cv2.putText(frame, f"Izq: {izq_num}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)
    cv2.putText(frame, f"Der: {der_num}", (400, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,0,0), 3)

    # Realizar operación
    resultado_texto = ""
    if izq_num > 0 and der_num > 0:
        if operacion == "suma":
            resultado = izq_num + der_num
            resultado_texto = f"{izq_num} + {der_num} = {resultado}"
        elif operacion == "resta":
            resultado = izq_num - der_num
            resultado_texto = f"{izq_num} - {der_num} = {resultado}"
        elif operacion == "multiplicacion":
            resultado = izq_num * der_num
            resultado_texto = f"{izq_num} x {der_num} = {resultado}"
        elif operacion == "division":
            if der_num != 0:
                resultado = izq_num / der_num
                resultado_texto = f"{izq_num} ÷ {der_num} = {resultado:.2f}"
            else:
                resultado_texto = f"{izq_num} ÷ {der_num} = ∞"

        cv2.putText(frame, resultado_texto, (150, 420), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0,255,0), 3)

    return frame, izq_num, der_num, resultado_texto

def frame_to_base64(frame):
    """Convierte un frame de OpenCV a base64"""
    _, buffer = cv2.imencode('.jpg', frame)
    frame_base64 = base64.b64encode(buffer).decode('utf-8')
    return frame_base64

class OperacionesMatematicas:
    def __init__(self):
        self.cap = None
        self.operacion_actual = "suma"
    
    def iniciar_camara(self):
        """Inicia la cámara"""
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                raise HTTPException(status_code=500, detail="No se pudo abrir la cámara")
    
    def detener_camara(self):
        """Detiene la cámara"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
    
    def obtener_frame(self, operacion: str = "suma"):
        """Obtiene un frame procesado con la operación especificada"""
        if self.cap is None:
            self.iniciar_camara()
        
        ret, frame = self.cap.read()
        if not ret:
            raise HTTPException(status_code=500, detail="No se pudo capturar frame")
        
        frame_procesado, izq_num, der_num, resultado = procesar_operacion(frame, operacion)
        frame_base64 = frame_to_base64(frame_procesado)
        
        return {
            "frame": frame_base64,
            "izquierda": izq_num,
            "derecha": der_num,
            "resultado": resultado,
            "operacion": operacion
        }
    
    def cambiar_operacion(self, operacion: str):
        """Cambia la operación actual"""
        operaciones_validas = ["suma", "resta", "multiplicacion", "division"]
        if operacion not in operaciones_validas:
            raise HTTPException(status_code=400, detail=f"Operación no válida. Use una de: {operaciones_validas}")
        
        self.operacion_actual = operacion
        return {"mensaje": f"Operación cambiada a: {operacion}", "operacion": operacion}

# Instancia global para manejar las operaciones
operaciones_handler = OperacionesMatematicas()

@router.get("/operaciones")
async def obtener_operaciones_disponibles():
    """Obtiene la lista de operaciones disponibles"""
    return {
        "operaciones": [
            {"id": "suma", "nombre": "Suma", "simbolo": "+", "descripcion": "Suma de dos números"},
            {"id": "resta", "nombre": "Resta", "simbolo": "-", "descripcion": "Resta de dos números"},
            {"id": "multiplicacion", "nombre": "Multiplicación", "simbolo": "×", "descripcion": "Multiplicación de dos números"},
            {"id": "division", "nombre": "División", "simbolo": "÷", "descripcion": "División de dos números"}
        ]
    }

@router.get("/frame")
async def obtener_frame_matematicas(operacion: str = "suma"):
    """Obtiene un frame procesado con operaciones matemáticas"""
    try:
        return operaciones_handler.obtener_frame(operacion)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/operacion")
async def cambiar_operacion_matematicas(operacion: str):
    """Cambia la operación matemática actual"""
    try:
        return operaciones_handler.cambiar_operacion(operacion)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Evento de shutdown se manejará en main.py
