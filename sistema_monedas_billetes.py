#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
PARCIAL DE VISIÓN ARTIFICIAL CLÁSICA - RETO: MONEDAS Y BILLETES DE EURO
================================================================================
Autor: Alumno de Visión Artificial
Descripción:
    Sistema experto de visión artificial clásica (sin Deep Learning) para la
    detección, clasificación mediante árbol de decisión explícito e interpretación
    contable de monedas (1c, 2c, 5c, 10c, 20c, 50c, 1€, 2€) y billetes (5€, 10€,
    20€, 50€) situados sobre un tapete verde mate de 0,50 x 0,50 m en vista cenital.

Cadena de procesamiento implementada:
    Adquisición RGB -> Preprocesamiento -> Segmentación HSV + Morfología ->
    Extracción de Características (Geométricas + Cromáticas Zonales) ->
    Clasificación por Árbol de Decisión -> Interpretación de Escena + HUD.
================================================================================
"""

import argparse
import base64
import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import cv2
import numpy as np


# ==============================================================================
# 1. PARÁMETROS DE CONFIGURACIÓN Y UMBRALES JUSTIFICADOS EXPERIMENTALMENTE
# ==============================================================================
@dataclass
class ConfiguracionSistema:
    """
    Centraliza todos los parámetros ópticos, morfológicos y los umbrales
    físicos/cromáticos del árbol de decisión para facilitar su ajuste y
    justificación experimental en el informe.
    """
    # --- Escala espacial (Calibración Cámara-Tapete) ---
    # Si el lado menor (ej. 1080 px) cubre 500 mm -> px_por_mm = 2.16 px/mm
    ancho_tapete_mm: float = 500.0
    px_por_mm: float = 2.16

    # --- Segmentación del tapete verde en espacio HSV (OpenCV: H en [0,180]) ---
    hsv_verde_min: Tuple[int, int, int] = (34, 40, 35)
    hsv_verde_max: Tuple[int, int, int] = (88, 255, 255)

    # --- Filtros de área física (en mm^2) para descartar ruido o intrusos ---
    # Moneda más pequeña: 1c (207.4 mm^2). Billete más grande: 50€ (10780 mm^2)
    area_min_objeto_mm2: float = 140.0
    area_max_objeto_mm2: float = 13500.0

    # --- Nodo Raíz: Geometría Global (Moneda vs Billete vs Solapamiento) ---
    umbral_circularidad_moneda: float = 0.78     # Círculo ideal = 1.0; Rectángulo ~ 0.62
    umbral_aspect_ratio_moneda: float = 1.22     # Moneda ~ 1.00 - 1.12
    umbral_solidez_minima: float = 0.91          # < 0.91 indica dos monedas tocándose
    aspect_ratio_billete_min: float = 1.65       # Billetes reales: 1.82 a 1.94
    aspect_ratio_billete_max: float = 2.15
    rectangularidad_billete_min: float = 0.84    # Área / (W_rot * H_rot)

    # --- Sub-árbol de Monedas: Análisis Zonal (Núcleo vs Corona en CIELAB) ---
    # En OpenCV Lab, a* y b* están desplazados +128 (neutro = 128.0)
    umbral_gradiente_bimetalico_b: float = 7.5   # |b*_nucleo - b*_corona| para 1€ y 2€
    umbral_area_1e_vs_2e_mm2: float = 472.7      # Punto medio entre 1€ (424.6) y 2€ (520.8)

    # Separación Cobre (1c, 2c, 5c) vs Oro Nórdico (10c, 20c, 50c)
    # El cobre tiene alta componente rojiza (a* > 136) y ratio (a*-128)/(b*-128) elevado
    umbral_cobre_lab_a: float = 136.5
    umbral_ratio_cobre_oro: float = 0.52

    # Umbrales de área (mm^2) óptimos (puntos medios entre clases de la misma familia)
    # Familia Cobriza: 1c (207.4), 2c (276.1), 5c (354.7)
    umbral_area_1c_2c_mm2: float = 241.7
    umbral_area_2c_5c_mm2: float = 315.4
    area_max_cobre_mm2: float = 415.0

    # Familia Dorada: 10c (306.4), 20c (388.8), 50c (461.9)
    area_min_oro_mm2: float = 260.0
    umbral_area_10c_20c_mm2: float = 347.6
    umbral_area_20c_50c_mm2: float = 425.3
    area_max_oro_mm2: float = 515.0

    # --- Sub-árbol de Billetes: Color HSV/Lab + Lado Mayor (mm) ---
    # 5€ (120x62mm, gris), 10€ (127x67mm, rojo), 20€ (133x72mm, azul), 50€ (140x77mm, naranja)
    umbral_saturacion_5e_gris: float = 48.0      # 5€ presenta saturación muy baja
    hue_azul_20e_min: int = 92                   # Rango H para azul en OpenCV (92 - 132)
    hue_azul_20e_max: int = 132
    umbral_lab_a_10e_vs_50e: float = 141.0       # 10€ (rojo) tiene mayor a* que 50€ (naranja)
    umbral_lado_10e_50e_mm: float = 133.5        # Punto medio entre 127 mm (10€) y 140 mm (50€)


# ==============================================================================
# 2. ESTRUCTURAS DE DATOS PARA CARACTERÍSTICAS Y RESULTADOS
# ==============================================================================
@dataclass
class CaracteristicasObjeto:
    """Almacena el vector explícito de características de un objeto segmentado."""
    id_objeto: int
    contorno: np.ndarray
    centroide_px: Tuple[int, int]
    caja_rotada: Tuple[Tuple[float, float], Tuple[float, float], float]

    # Descriptores geométricos
    area_px: float
    area_mm2: float
    perimetro_mm: float
    circularidad: float
    lado_mayor_mm: float
    lado_menor_mm: float
    relacion_aspecto: float
    rectangularidad: float
    solidez: float
    orientacion_grados: float

    # Descriptores cromáticos y zonales (Núcleo y Corona)
    hue_mediana: float
    sat_mediana: float
    val_mediana: float
    lab_a_interior: float
    lab_b_interior: float
    lab_b_nucleo: float
    lab_b_corona: float
    delta_b_radial: float
    ratio_cobre_oro: float


@dataclass
class ResultadoClasificacion:
    """Resultado emitido por el árbol de decisión para un objeto."""
    id_objeto: int
    categoria: str                 # 'MONEDA', 'BILLETE', 'SOLAPAMIENTO', 'INTRUSO'
    etiqueta: str                  # Ej: '2.00 EUR', '20.00 EUR', 'INTRUSO'
    valor_centimos: int            # Valor monetario en céntimos de euro
    ruta_arbol: List[str] = field(default_factory=list)
    caracteristicas: Optional[CaracteristicasObjeto] = None


# ==============================================================================
# 3. ETAPA 1 Y 2: PREPROCESAMIENTO Y SEGMENTACIÓN CLÁSICA
# ==============================================================================
def segmentar_escena(
    img_bgr: np.ndarray,
    cfg: ConfiguracionSistema
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Aplica preprocesamiento, segmentación por color en espacio HSV para aislar
    el fondo del tapete verde mate, filtrado morfológico y relleno de huecos.

    Parámetros:
        img_bgr: Imagen de entrada en formato BGR (OpenCV).
        cfg: Configuración del sistema con umbrales HSV y escala.

    Retorna:
        mascara_objetos: Máscara binaria depurada con los objetos en blanco (255).
        contornos_validos: Lista de contornos externos filtrados por área mínima.
    """
    # 1. Suavizado Gaussiano para atenuar la textura fibrosa del fieltro verde
    img_suavizada = cv2.GaussianBlur(img_bgr, (5, 5), 0)

    # 2. Conversión al espacio HSV y umbralización del tapete verde
    img_hsv = cv2.cvtColor(img_suavizada, cv2.COLOR_BGR2HSV)
    mascara_fondo_verde = cv2.inRange(
        img_hsv,
        np.array(cfg.hsv_verde_min, dtype=np.uint8),
        np.array(cfg.hsv_verde_max, dtype=np.uint8)
    )

    # 3. Inversión de la máscara: los objetos son todo aquello que NO es tapete
    mascara_cruda = cv2.bitwise_not(mascara_fondo_verde)

    # 4. Operaciones morfológicas:
    #    - Apertura (OPEN) para eliminar motas blancas de polvo sobre el tapete
    #    - Cierre (CLOSE) para unir grietas producidas por reflejos o bandas holográficas
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mascara_limpia = cv2.morphologyEx(mascara_cruda, cv2.MORPH_OPEN, kernel_open)
    mascara_limpia = cv2.morphologyEx(mascara_limpia, cv2.MORPH_CLOSE, kernel_close)

    # 5. Extracción de contornos exteriores y relleno sólido de huecos internos
    contornos, _ = cv2.findContours(
        mascara_limpia, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    mascara_objetos = np.zeros_like(mascara_limpia)
    contornos_validos = []
    escala_area = cfg.px_por_mm ** 2

    for cnt in contornos:
        area_px = cv2.contourArea(cnt)
        area_mm2 = area_px / escala_area

        # Filtrado inicial de ruido por debajo de 140 mm^2 (moneda 1c = 207.4 mm^2)
        if cfg.area_min_objeto_mm2 <= area_mm2 <= cfg.area_max_objeto_mm2:
            contornos_validos.append(cnt)
            cv2.drawContours(mascara_objetos, [cnt], -1, 255, thickness=cv2.FILLED)

    return mascara_objetos, contornos_validos


# ==============================================================================
# 4. ETAPA 3: EXTRACCIÓN DE CARACTERÍSTICAS GEOMÉTRICAS Y CROMÁTICAS ZONALES
# ==============================================================================
def extraer_caracteristicas(
    img_bgr: np.ndarray,
    contorno: np.ndarray,
    id_obj: int,
    cfg: ConfiguracionSistema
) -> CaracteristicasObjeto:
    """
    Calcula descriptores geométricos invariantes a traslación y rotación,
    así como descriptores cromáticos zonales (núcleo vs corona) en HSV y CIELAB.
    """
    escala_lin = cfg.px_por_mm
    escala_area = escala_lin ** 2

    # --- A. Descriptores Geométricos ---
    area_px = float(cv2.contourArea(contorno))
    area_mm2 = area_px / escala_area

    perimetro_px = float(cv2.arcLength(contorno, True))
    perimetro_mm = perimetro_px / escala_lin

    # Circularidad de Cox: C = 4 * pi * A / P^2 (1.0 en círculo perfecto)
    circularidad = (4.0 * np.pi * area_px) / (perimetro_px ** 2 + 1e-6)

    # Momentos y Centroide
    momentos = cv2.moments(contorno)
    cx = int(momentos["m10"] / (momentos["m00"] + 1e-6))
    cy = int(momentos["m01"] / (momentos["m00"] + 1e-6))

    # Caja delimitadora rotada de área mínima (invariante a la orientación)
    caja_rotada = cv2.minAreaRect(contorno)
    (_, _), (w_rot_px, h_rot_px), angulo = caja_rotada
    lado_mayor_px = max(w_rot_px, h_rot_px)
    lado_menor_px = max(min(w_rot_px, h_rot_px), 1.0)

    lado_mayor_mm = lado_mayor_px / escala_lin
    lado_menor_mm = lado_menor_px / escala_lin
    relacion_aspecto = lado_mayor_px / lado_menor_px
    rectangularidad = area_px / (lado_mayor_px * lado_menor_px + 1e-6)

    # Envolvente convexa y Solidez (detecta concavidades por monedas en contacto)
    hull = cv2.convexHull(contorno)
    area_hull = float(cv2.contourArea(hull))
    solidez = area_px / (area_hull + 1e-6)

    # --- B. Descriptores Cromáticos Zonales sobre recorte (ROI) para eficiencia ---
    x, y, w, h = cv2.boundingRect(contorno)
    roi_bgr = img_bgr[y:y+h, x:x+w]
    roi_hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    roi_lab = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2Lab)

    # Desplazar el contorno a coordenadas locales del ROI
    cnt_local = contorno - np.array([[x, y]])
    mascara_local = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(mascara_local, [cnt_local], -1, 255, thickness=cv2.FILLED)

    # Erosión del 12% para evitar píxeles contaminados por el borde del tapete verde
    r_eq_px = np.sqrt(area_px / np.pi)
    k_erosion = max(3, int(0.12 * r_eq_px) | 1)
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_erosion, k_erosion))
    mascara_interior = cv2.erode(mascara_local, kernel_erode)
    if np.count_nonzero(mascara_interior) < 10:
        mascara_interior = mascara_local

    # Construcción de máscaras concéntricas (Núcleo y Corona) respecto al centroide local
    cx_local, cy_local = cx - x, cy - y
    yy, xx = np.ogrid[:h, :w]
    dist_centro = np.sqrt((xx - cx_local) ** 2 + (yy - cy_local) ** 2)

    mascara_nucleo = ((dist_centro <= 0.50 * r_eq_px) & (mascara_local > 0)).astype(np.uint8) * 255
    mascara_corona = (
        (dist_centro >= 0.64 * r_eq_px) &
        (dist_centro <= 0.88 * r_eq_px) &
        (mascara_local > 0)
    ).astype(np.uint8) * 255

    # Extracción robusta mediante medianas en la región interior
    pix_hsv = roi_hsv[mascara_interior > 0]
    pix_lab = roi_lab[mascara_interior > 0]

    hue_med = float(np.median(pix_hsv[:, 0]))
    sat_med = float(np.median(pix_hsv[:, 1]))
    val_med = float(np.median(pix_hsv[:, 2]))

    lab_a_int = float(np.median(pix_lab[:, 1]))
    lab_b_int = float(np.median(pix_lab[:, 2]))

    # Cálculo de cromaticidad amarilla-azul (b*) en Núcleo y Corona para monedas bimetálicas
    pix_lab_nuc = roi_lab[mascara_nucleo > 0]
    pix_lab_cor = roi_lab[mascara_corona > 0]

    lab_b_nuc = float(np.median(pix_lab_nuc[:, 2])) if len(pix_lab_nuc) > 5 else lab_b_int
    lab_b_cor = float(np.median(pix_lab_cor[:, 2])) if len(pix_lab_cor) > 5 else lab_b_int

    # Gradiente radial: positivo si el núcleo es más dorado que la corona (2€),
    # negativo si la corona es más dorada que el núcleo (1€), ~0 en monocolor.
    delta_b_radial = lab_b_nuc - lab_b_cor

    # Índice Cobre vs Oro: cociente de componente rojiza (a*-128) frente a amarilla (b*-128)
    a_neto = max(0.0, lab_a_int - 128.0)
    b_neto = max(1.0, lab_b_int - 128.0)
    ratio_cobre_oro = a_neto / b_neto

    return CaracteristicasObjeto(
        id_objeto=id_obj,
        contorno=contorno,
        centroide_px=(cx, cy),
        caja_rotada=caja_rotada,
        area_px=area_px,
        area_mm2=area_mm2,
        perimetro_mm=perimetro_mm,
        circularidad=circularidad,
        lado_mayor_mm=lado_mayor_mm,
        lado_menor_mm=lado_menor_mm,
        relacion_aspecto=relacion_aspecto,
        rectangularidad=rectangularidad,
        solidez=solidez,
        orientacion_grados=float(angulo),
        hue_mediana=hue_med,
        sat_mediana=sat_med,
        val_mediana=val_med,
        lab_a_interior=lab_a_int,
        lab_b_interior=lab_b_int,
        lab_b_nucleo=lab_b_nuc,
        lab_b_corona=lab_b_cor,
        delta_b_radial=delta_b_radial,
        ratio_cobre_oro=ratio_cobre_oro
    )


# ==============================================================================
# 5. ETAPA 4: ÁRBOL DE DECISIÓN EXPLÍCITO Y TRAZABLE
# ==============================================================================
def clasificar_objeto_arbol(
    f: CaracteristicasObjeto,
    cfg: ConfiguracionSistema
) -> ResultadoClasificacion:
    """
    Árbol de decisión jerárquico basado en reglas explícitas sobre características
    geométricas y cromáticas. Registra cada nodo atravesado en `ruta_arbol`.
    """
    ruta = []

    # --------------------------------------------------------------------------
    # NODO 1 (RAÍZ): Separación Morfológica Global (Moneda vs Billete vs Anomalía)
    # --------------------------------------------------------------------------
    if f.circularidad >= cfg.umbral_circularidad_moneda and f.relacion_aspecto <= cfg.umbral_aspect_ratio_moneda:
        ruta.append(f"N1: Circularidad={f.circularidad:.2f}>={cfg.umbral_circularidad_moneda} -> MONEDA")

        # ----------------------------------------------------------------------
        # NODO 2 (MONEDAS): Contraste Radial Núcleo-Corona (¿Es Bimetálica?)
        # ----------------------------------------------------------------------
        if abs(f.delta_b_radial) >= cfg.umbral_gradiente_bimetalico_b and f.area_mm2 >= 380.0:
            ruta.append(f"N2: |Δb*_radial|={abs(f.delta_b_radial):.1f}>={cfg.umbral_gradiente_bimetalico_b} -> BIMETÁLICA")

            # NODO 3A: Signo del gradiente radial + Verificación de Área (1€ vs 2€)
            if f.delta_b_radial < 0 and f.area_mm2 < cfg.umbral_area_1e_vs_2e_mm2:
                ruta.append(f"N3A: Δb*<0 (Corona Oro) y A={f.area_mm2:.0f}<{cfg.umbral_area_1e_vs_2e_mm2:.0f} -> 1€")
                return ResultadoClasificacion(f.id_objeto, "MONEDA", "1.00 EUR", 100, ruta, f)
            elif f.delta_b_radial > 0 and f.area_mm2 >= cfg.umbral_area_1e_vs_2e_mm2:
                ruta.append(f"N3A: Δb*>0 (Núcleo Oro) y A={f.area_mm2:.0f}>={cfg.umbral_area_1e_vs_2e_mm2:.0f} -> 2€")
                return ResultadoClasificacion(f.id_objeto, "MONEDA", "2.00 EUR", 200, ruta, f)
            else:
                # Desempate por área si un reflejo atenúa parcialmente el signo
                if f.area_mm2 < cfg.umbral_area_1e_vs_2e_mm2:
                    ruta.append(f"N3A(Aux): A={f.area_mm2:.0f}<{cfg.umbral_area_1e_vs_2e_mm2:.0f} -> 1€")
                    return ResultadoClasificacion(f.id_objeto, "MONEDA", "1.00 EUR", 100, ruta, f)
                else:
                    ruta.append(f"N3A(Aux): A={f.area_mm2:.0f}>={cfg.umbral_area_1e_vs_2e_mm2:.0f} -> 2€")
                    return ResultadoClasificacion(f.id_objeto, "MONEDA", "2.00 EUR", 200, ruta, f)

        else:
            ruta.append(f"N2: |Δb*_radial|={abs(f.delta_b_radial):.1f}<{cfg.umbral_gradiente_bimetalico_b} -> MONOCOLOR")

            # ------------------------------------------------------------------
            # NODO 3B: Familia Cromática (Acero Cobrizo vs Oro Nórdico)
            # ------------------------------------------------------------------
            es_cobre = (
                f.lab_a_interior >= cfg.umbral_cobre_lab_a or
                f.ratio_cobre_oro >= cfg.umbral_ratio_cobre_oro
            )

            if es_cobre:
                ruta.append(f"N3B: a*={f.lab_a_interior:.1f} / Ratio={f.ratio_cobre_oro:.2f} -> FAMILIA COBRE")

                # NODO 4A: Clasificación por Área dentro de Familia Cobre (1c, 2c, 5c)
                if f.area_mm2 < cfg.umbral_area_1c_2c_mm2:
                    ruta.append(f"N4A: A={f.area_mm2:.0f}<{cfg.umbral_area_1c_2c_mm2:.0f}mm² -> 1 céntimo")
                    return ResultadoClasificacion(f.id_objeto, "MONEDA", "0.01 EUR", 1, ruta, f)
                elif f.area_mm2 < cfg.umbral_area_2c_5c_mm2:
                    ruta.append(f"N4A: A={f.area_mm2:.0f}<{cfg.umbral_area_2c_5c_mm2:.0f}mm² -> 2 céntimos")
                    return ResultadoClasificacion(f.id_objeto, "MONEDA", "0.02 EUR", 2, ruta, f)
                elif f.area_mm2 <= cfg.area_max_cobre_mm2:
                    ruta.append(f"N4A: A={f.area_mm2:.0f}>={cfg.umbral_area_2c_5c_mm2:.0f}mm² -> 5 céntimos")
                    return ResultadoClasificacion(f.id_objeto, "MONEDA", "0.05 EUR", 5, ruta, f)
                else:
                    ruta.append(f"N4A: Área fuera de rango cobre ({f.area_mm2:.0f}mm²) -> INTRUSO")
                    return ResultadoClasificacion(f.id_objeto, "INTRUSO", "INTRUSO (Cobre > 5c)", 0, ruta, f)

            else:
                ruta.append(f"N3B: b*={f.lab_b_interior:.1f} dominante -> FAMILIA ORO NÓRDICO")

                # NODO 4B: Clasificación por Área dentro de Familia Oro (10c, 20c, 50c)
                if f.area_mm2 < cfg.area_min_oro_mm2 or f.area_mm2 > cfg.area_max_oro_mm2:
                    ruta.append(f"N4B: Área fuera de rango oro ({f.area_mm2:.0f}mm²) -> INTRUSO")
                    return ResultadoClasificacion(f.id_objeto, "INTRUSO", "INTRUSO (Circular)", 0, ruta, f)
                elif f.area_mm2 < cfg.umbral_area_10c_20c_mm2:
                    ruta.append(f"N4B: A={f.area_mm2:.0f}<{cfg.umbral_area_10c_20c_mm2:.0f}mm² -> 10 céntimos")
                    return ResultadoClasificacion(f.id_objeto, "MONEDA", "0.10 EUR", 10, ruta, f)
                elif f.area_mm2 < cfg.umbral_area_20c_50c_mm2:
                    ruta.append(f"N4B: A={f.area_mm2:.0f}<{cfg.umbral_area_20c_50c_mm2:.0f}mm² -> 20 céntimos")
                    return ResultadoClasificacion(f.id_objeto, "MONEDA", "0.20 EUR", 20, ruta, f)
                else:
                    ruta.append(f"N4B: A={f.area_mm2:.0f}>={cfg.umbral_area_20c_50c_mm2:.0f}mm² -> 50 céntimos")
                    return ResultadoClasificacion(f.id_objeto, "MONEDA", "0.50 EUR", 50, ruta, f)

    # --------------------------------------------------------------------------
    # RAMA DERECHA NODO 1: ¿Es un Billete Rectangular o son Monedas Solapadas?
    # --------------------------------------------------------------------------
    es_geometria_billete = (
        cfg.aspect_ratio_billete_min <= f.relacion_aspecto <= cfg.aspect_ratio_billete_max and
        f.rectangularidad >= cfg.rectangularidad_billete_min and
        f.area_mm2 >= 5500.0
    )

    if es_geometria_billete:
        ruta.append(
            f"N1: AR={f.relacion_aspecto:.2f}, Rect={f.rectangularidad:.2f}, "
            f"A={f.area_mm2:.0f}mm² -> BILLETE"
        )

        # ----------------------------------------------------------------------
        # NODO 2 (BILLETES): Saturación (Billete Gris de 5€ vs Billetes Coloreados)
        # ----------------------------------------------------------------------
        if f.sat_mediana < cfg.umbral_saturacion_5e_gris and f.lado_mayor_mm < 124.5:
            ruta.append(f"N2_B: Sat={f.sat_mediana:.1f}<{cfg.umbral_saturacion_5e_gris} y L={f.lado_mayor_mm:.1f}mm -> 5€")
            return ResultadoClasificacion(f.id_objeto, "BILLETE", "5.00 EUR", 500, ruta, f)

        # ----------------------------------------------------------------------
        # NODO 3 (BILLETES): Tono Azul (20€) vs Cálidos (10€ Rojo / 50€ Naranja)
        # ----------------------------------------------------------------------
        if cfg.hue_azul_20e_min <= f.hue_mediana <= cfg.hue_azul_20e_max:
            ruta.append(f"N3_B: Hue={f.hue_mediana:.0f} en rango Azul [92,132] y L={f.lado_mayor_mm:.1f}mm -> 20€")
            return ResultadoClasificacion(f.id_objeto, "BILLETE", "20.00 EUR", 2000, ruta, f)

        # ----------------------------------------------------------------------
        # NODO 4 (BILLETES): Cálidos -> 10€ (Rojo, 127mm) vs 50€ (Naranja, 140mm)
        # ----------------------------------------------------------------------
        if f.lado_mayor_mm < cfg.umbral_lado_10e_50e_mm and f.lab_a_interior >= cfg.umbral_lab_a_10e_vs_50e:
            ruta.append(f"N4_B: L={f.lado_mayor_mm:.1f}mm<133.5 y a*={f.lab_a_interior:.1f} (Rojo) -> 10€")
            return ResultadoClasificacion(f.id_objeto, "BILLETE", "10.00 EUR", 1000, ruta, f)
        elif f.lado_mayor_mm >= cfg.umbral_lado_10e_50e_mm:
            ruta.append(f"N4_B: L={f.lado_mayor_mm:.1f}mm>=133.5 (Naranja/Ocre) -> 50€")
            return ResultadoClasificacion(f.id_objeto, "BILLETE", "50.00 EUR", 5000, ruta, f)
        else:
            # Desempate por longitud si la iluminación altera levemente el canal a*
            if f.lado_mayor_mm < 123.5:
                ruta.append(f"N4_B(Aux): L={f.lado_mayor_mm:.1f}mm -> 5€")
                return ResultadoClasificacion(f.id_objeto, "BILLETE", "5.00 EUR", 500, ruta, f)
            elif f.lado_mayor_mm < cfg.umbral_lado_10e_50e_mm:
                ruta.append(f"N4_B(Aux): L={f.lado_mayor_mm:.1f}mm -> 10€")
                return ResultadoClasificacion(f.id_objeto, "BILLETE", "10.00 EUR", 1000, ruta, f)
            else:
                ruta.append(f"N4_B(Aux): L={f.lado_mayor_mm:.1f}mm -> 50€")
                return ResultadoClasificacion(f.id_objeto, "BILLETE", "50.00 EUR", 5000, ruta, f)

    # --------------------------------------------------------------------------
    # DETECCIÓN DE SOLAPAMIENTO O ELEMENTOS INTRUSOS (Nivel 3 de Interpretación)
    # --------------------------------------------------------------------------
    if f.solidez < cfg.umbral_solidez_minima and f.area_mm2 < 2500.0:
        ruta.append(f"N1_Anomalía: Solidez={f.solidez:.2f}<{cfg.umbral_solidez_minima} -> MONEDAS EN CONTACTO")
        return ResultadoClasificacion(f.id_objeto, "SOLAPAMIENTO", "ALERTA: CONTACTO", 0, ruta, f)

    ruta.append(f"N1_Anomalía: C={f.circularidad:.2f}, AR={f.relacion_aspecto:.2f} no válido -> INTRUSO")
    return ResultadoClasificacion(f.id_objeto, "INTRUSO", "INTRUSO / NO VÁLIDO", 0, ruta, f)


# ==============================================================================
# 6. CALIBRACIÓN DE ESCALA (PX/MM) CON CUALQUIER MONEDA O BILLETE
# ==============================================================================
# Catálogo oficial de dimensiones físicas (Banco Central Europeo)
REFERENCIAS_CALIBRACION: Dict[str, Dict[str, float | str]] = {
    # Monedas: diámetro oficial en mm
    "Moneda 0.01 EUR (1 céntimo - 16.25 mm)":  {"tipo": "MONEDA", "etiqueta": "0.01 EUR", "diametro_mm": 16.25},
    "Moneda 0.02 EUR (2 céntimos - 18.75 mm)": {"tipo": "MONEDA", "etiqueta": "0.02 EUR", "diametro_mm": 18.75},
    "Moneda 0.05 EUR (5 céntimos - 21.25 mm)": {"tipo": "MONEDA", "etiqueta": "0.05 EUR", "diametro_mm": 21.25},
    "Moneda 0.10 EUR (10 céntimos - 19.75 mm)": {"tipo": "MONEDA", "etiqueta": "0.10 EUR", "diametro_mm": 19.75},
    "Moneda 0.20 EUR (20 céntimos - 22.25 mm)": {"tipo": "MONEDA", "etiqueta": "0.20 EUR", "diametro_mm": 22.25},
    "Moneda 0.50 EUR (50 céntimos - 24.25 mm)": {"tipo": "MONEDA", "etiqueta": "0.50 EUR", "diametro_mm": 24.25},
    "Moneda 1.00 EUR (1 euro - 23.25 mm)":      {"tipo": "MONEDA", "etiqueta": "1.00 EUR", "diametro_mm": 23.25},
    "Moneda 2.00 EUR (2 euros - 25.75 mm)":     {"tipo": "MONEDA", "etiqueta": "2.00 EUR", "diametro_mm": 25.75},
    # Billetes: largo x ancho oficiales en mm
    "Billete 5.00 EUR (5 euros - 120x62 mm)":   {"tipo": "BILLETE", "etiqueta": "5.00 EUR",  "largo_mm": 120.0, "ancho_mm": 62.0},
    "Billete 10.00 EUR (10 euros - 127x67 mm)": {"tipo": "BILLETE", "etiqueta": "10.00 EUR", "largo_mm": 127.0, "ancho_mm": 67.0},
    "Billete 20.00 EUR (20 euros - 133x72 mm)": {"tipo": "BILLETE", "etiqueta": "20.00 EUR", "largo_mm": 133.0, "ancho_mm": 72.0},
    "Billete 50.00 EUR (50 euros - 140x77 mm)": {"tipo": "BILLETE", "etiqueta": "50.00 EUR", "largo_mm": 140.0, "ancho_mm": 77.0},
}


def obtener_contornos_candidatos_calibracion(
    img_bgr: np.ndarray,
    cfg: ConfiguracionSistema
) -> List[np.ndarray]:
    """
    Extrae todos los contornos de la escena sin depender de la escala px/mm actual,
    para poder calibrar incluso si la escala previa era totalmente errónea.
    """
    img_suavizada = cv2.GaussianBlur(img_bgr, (5, 5), 0)
    img_hsv = cv2.cvtColor(img_suavizada, cv2.COLOR_BGR2HSV)
    mascara_fondo_verde = cv2.inRange(
        img_hsv,
        np.array(cfg.hsv_verde_min, dtype=np.uint8),
        np.array(cfg.hsv_verde_max, dtype=np.uint8)
    )
    mascara_cruda = cv2.bitwise_not(mascara_fondo_verde)
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mascara_limpia = cv2.morphologyEx(mascara_cruda, cv2.MORPH_OPEN, kernel_open)
    mascara_limpia = cv2.morphologyEx(mascara_limpia, cv2.MORPH_CLOSE, kernel_close)

    contornos, _ = cv2.findContours(
        mascara_limpia, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    area_img = img_bgr.shape[0] * img_bgr.shape[1]
    candidatos = [
        cnt for cnt in contornos
        if 120.0 <= cv2.contourArea(cnt) <= 0.85 * area_img
    ]
    return candidatos


def calcular_escala_desde_contorno(
    contorno: np.ndarray,
    info_ref: Dict[str, float | str]
) -> float:
    """
    Calcula la relación px/mm a partir de un contorno específico y los datos
    físicos oficiales de la moneda o billete elegido como referencia.
    """
    tipo = str(info_ref["tipo"])
    if tipo == "MONEDA":
        diametro_ref_mm = float(info_ref["diametro_mm"])
        area_px = float(cv2.contourArea(contorno))
        diametro_medido_px = 2.0 * np.sqrt(area_px / np.pi)
        return float(diametro_medido_px / diametro_ref_mm)
    else:
        largo_ref_mm = float(info_ref["largo_mm"])
        ancho_ref_mm = float(info_ref["ancho_mm"])
        (_, _), (w_rot_px, h_rot_px), _ = cv2.minAreaRect(contorno)
        lado_mayor_px = max(w_rot_px, h_rot_px)
        lado_menor_px = max(min(w_rot_px, h_rot_px), 1.0)
        # Promedio entre escala longitudinal y transversal del billete para mayor robustez
        escala_largo = lado_mayor_px / largo_ref_mm
        escala_ancho = lado_menor_px / ancho_ref_mm
        return float(0.5 * (escala_largo + escala_ancho))


def autocalibrar_con_referencia(
    img_bgr: np.ndarray,
    cfg: ConfiguracionSistema,
    clave_referencia: str,
    contorno_especifico: Optional[np.ndarray] = None,
    criterio_auto: str = "MAYOR"
) -> float:
    """
    Calibra `cfg.px_por_mm` usando cualquier moneda (1c a 2€) o billete (5€ a 50€).
    - Si se proporciona `contorno_especifico`, calibra directamente con ese objeto.
    - Si no, busca automáticamente entre los objetos con geometría compatible
      (circular para monedas, rectangular para billetes) según `criterio_auto`
      ('MAYOR' o 'MENOR').
    """
    if clave_referencia not in REFERENCIAS_CALIBRACION:
        raise ValueError(f"Referencia no reconocida: {clave_referencia}")

    info_ref = REFERENCIAS_CALIBRACION[clave_referencia]
    tipo_esperado = str(info_ref["tipo"])

    if contorno_especifico is not None:
        nuevo_px_por_mm = calcular_escala_desde_contorno(contorno_especifico, info_ref)
        cfg.px_por_mm = nuevo_px_por_mm
        return nuevo_px_por_mm

    candidatos = obtener_contornos_candidatos_calibracion(img_bgr, cfg)
    compatibles: List[Tuple[float, np.ndarray]] = []

    for cnt in candidatos:
        area_px = float(cv2.contourArea(cnt))
        perim_px = float(cv2.arcLength(cnt, True))
        circ = (4.0 * np.pi * area_px) / (perim_px ** 2 + 1e-6)
        (_, _), (w_px, h_px), _ = cv2.minAreaRect(cnt)
        lado_mayor = max(w_px, h_px)
        lado_menor = max(min(w_px, h_px), 1.0)
        ar = lado_mayor / lado_menor
        rect = area_px / (lado_mayor * lado_menor + 1e-6)

        if tipo_esperado == "MONEDA":
            if circ >= 0.72 and ar <= 1.28:
                compatibles.append((area_px, cnt))
        else:
            if 1.50 <= ar <= 2.30 and rect >= 0.78:
                compatibles.append((area_px, cnt))

    if not compatibles:
        nombre_tipo = "monedas circulares" if tipo_esperado == "MONEDA" else "billetes rectangulares"
        raise ValueError(f"No se encontraron {nombre_tipo} en la escena para calibrar.")

    compatibles.sort(key=lambda item: item[0])
    contorno_elegido = compatibles[-1][1] if criterio_auto == "MAYOR" else compatibles[0][1]

    nuevo_px_por_mm = calcular_escala_desde_contorno(contorno_elegido, info_ref)
    cfg.px_por_mm = nuevo_px_por_mm
    return nuevo_px_por_mm


def autocalibrar_por_moneda_mayor(
    img_bgr: np.ndarray,
    cfg: ConfiguracionSistema,
    diametro_ref_mm: float = 25.75
) -> float:
    """Función de compatibilidad para calibración rápida con moneda mayor."""
    _, contornos = segmentar_escena(img_bgr, cfg)
    if not contornos:
        contornos = obtener_contornos_candidatos_calibracion(img_bgr, cfg)
    areas_circulares = []
    for cnt in contornos:
        area_px = cv2.contourArea(cnt)
        perim_px = cv2.arcLength(cnt, True)
        circ = (4.0 * np.pi * area_px) / (perim_px ** 2 + 1e-6)
        if circ >= 0.75:
            areas_circulares.append(area_px)

    if not areas_circulares:
        raise ValueError("No se encontraron monedas circulares para autocalibrar.")

    area_max_px = max(areas_circulares)
    diametro_medido_px = 2.0 * np.sqrt(area_max_px / np.pi)
    nuevo_px_por_mm = float(diametro_medido_px / diametro_ref_mm)
    cfg.px_por_mm = nuevo_px_por_mm
    return nuevo_px_por_mm


# ==============================================================================
# 7. ETAPA 5: INTERPRETACIÓN DE LA ESCENA Y VISUALIZACIÓN HUD
# ==============================================================================
def interpretar_y_visualizar(
    img_bgr: np.ndarray,
    resultados: List[ResultadoClasificacion],
    cfg: ConfiguracionSistema
) -> Tuple[np.ndarray, Dict]:
    """
    Realiza el análisis de Nivel 3 (conteo por denominación, suma monetaria exacta,
    detección de anomalías) y dibuja un panel HUD sobre la escena.
    """
    canvas = img_bgr.copy()
    total_centimos = 0
    num_monedas = 0
    num_billetes = 0
    num_alertas = 0
    desglose: Dict[str, int] = {}

    colores_cat = {
        "MONEDA": (0, 230, 255),       # Amarillo/Dorado en BGR
        "BILLETE": (255, 205, 50),     # Azul claro en BGR
        "SOLAPAMIENTO": (0, 140, 255), # Naranja alerta
        "INTRUSO": (40, 40, 255)       # Rojo puro
    }

    for res in resultados:
        f = res.caracteristicas
        color = colores_cat.get(res.categoria, (255, 255, 255))

        # Contabilidad de Nivel 3
        total_centimos += res.valor_centimos
        if res.categoria == "MONEDA":
            num_monedas += 1
            desglose[res.etiqueta] = desglose.get(res.etiqueta, 0) + 1
        elif res.categoria == "BILLETE":
            num_billetes += 1
            desglose[res.etiqueta] = desglose.get(res.etiqueta, 0) + 1
        else:
            num_alertas += 1

        cx, cy = f.centroide_px

        # Dibujo geométrico según el tipo de objeto
        if res.categoria == "MONEDA":
            r_px = int(np.sqrt(f.area_px / np.pi))
            cv2.circle(canvas, (cx, cy), r_px, color, 2, cv2.LINE_AA)
            # Anillo de análisis zonal (núcleo/corona)
            cv2.circle(canvas, (cx, cy), int(0.50 * r_px), (210, 210, 210), 1, cv2.LINE_AA)
            y_label = cy - r_px - 18
        else:
            box = np.int32(cv2.boxPoints(f.caja_rotada))
            cv2.drawContours(canvas, [box], 0, color, 2, cv2.LINE_AA)
            y_label = cy - 8

        cv2.circle(canvas, (cx, cy), 3, (0, 0, 255), -1)

        texto_principal = f"#{res.id_objeto} {res.etiqueta}"
        texto_secundario = f"{f.area_mm2:.0f}mm2 | C:{f.circularidad:.2f}"
        cv2.putText(
            canvas, texto_principal, (cx - 48, y_label),
            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 3, cv2.LINE_AA
        )
        cv2.putText(
            canvas, texto_principal, (cx - 48, y_label),
            cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA
        )
        cv2.putText(
            canvas, texto_secundario, (cx - 48, y_label + 16),
            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (235, 235, 235), 1, cv2.LINE_AA
        )

    # --- Panel HUD Superior de Interpretación de la Escena ---
    alto_hud = 128
    overlay = canvas.copy()
    cv2.rectangle(overlay, (10, 10), (canvas.shape[1] - 10, alto_hud), (18, 26, 18), -1)
    cv2.addWeighted(overlay, 0.84, canvas, 0.16, 0, canvas)
    cv2.rectangle(canvas, (10, 10), (canvas.shape[1] - 10, alto_hud), (0, 220, 120), 1)

    total_euros = total_centimos / 100.0
    cv2.putText(
        canvas,
        f"IMPORTE TOTAL: {total_euros:.2f} EUR",
        (25, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 140), 2, cv2.LINE_AA
    )
    cv2.putText(
        canvas,
        f"Monedas: {num_monedas} | Billetes: {num_billetes} | Alertas/Intrusos: {num_alertas} | Escala: {cfg.px_por_mm:.2f} px/mm",
        (25, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (225, 225, 225), 1, cv2.LINE_AA
    )

    # Formateo compacto del inventario para evitar que se corte en pantalla
    items_inv = [f"{k.replace(' EUR', 'E')}: x{v}" for k, v in sorted(desglose.items())]
    mitad = (len(items_inv) + 1) // 2
    linea_inv_1 = "Inventario: " + (" | ".join(items_inv[:mitad]) if items_inv else "Vacío")
    linea_inv_2 = "            " + (" | ".join(items_inv[mitad:]) if len(items_inv) > mitad else "")

    cv2.putText(
        canvas, linea_inv_1, (25, 95),
        cv2.FONT_HERSHEY_SIMPLEX, 0.46, (180, 235, 255), 1, cv2.LINE_AA
    )
    if linea_inv_2.strip():
        cv2.putText(
            canvas, linea_inv_2, (25, 116),
            cv2.FONT_HERSHEY_SIMPLEX, 0.46, (180, 235, 255), 1, cv2.LINE_AA
        )

    resumen = {
        "total_euros": total_euros,
        "total_centimos": total_centimos,
        "num_monedas": num_monedas,
        "num_billetes": num_billetes,
        "num_alertas": num_alertas,
        "desglose": desglose
    }
    return canvas, resumen


# ==============================================================================
# 8. GENERADOR DE ESCENA SINTÉTICA DE PRUEBA (PARA VERIFICACIÓN INMEDIATA)
# ==============================================================================
def generar_escena_sintetica_prueba(cfg: ConfiguracionSistema) -> np.ndarray:
    """
    Crea una escena cenital sintética a escala exacta sobre un tapete verde de
    500 x 500 mm con las 8 monedas y los 4 billetes rotados en ángulos arbitrarios
    para validar el árbol de decisión de forma inmediata.
    """
    lado_px = int(cfg.ancho_tapete_mm * cfg.px_por_mm)
    # Tapete verde mate en BGR (H ~ 60 en HSV)
    escena = np.full((lado_px, lado_px, 3), (48, 115, 45), dtype=np.uint8)

    # Textura fibrosa simulada del fieltro
    np.random.seed(42)
    ruido = np.random.normal(0, 3.0, escena.shape).astype(np.int16)
    escena = np.clip(escena.astype(np.int16) + ruido, 0, 255).astype(np.uint8)

    s = cfg.px_por_mm

    def dibujar_billete_rotado(centro_mm, dims_mm, angulo, color_bgr):
        cx, cy = int(centro_mm[0] * s), int(centro_mm[1] * s)
        w, h = dims_mm[0] * s, dims_mm[1] * s
        rect = ((cx, cy), (w, h), angulo)
        box = np.int32(cv2.boxPoints(rect))
        cv2.drawContours(escena, [box], 0, color_bgr, -1, cv2.LINE_AA)

    def dibujar_moneda(centro_mm, diametro_mm, color_corona_bgr, color_nucleo_bgr=None):
        cx, cy = int(centro_mm[0] * s), int(centro_mm[1] * s)
        r_ext = int((diametro_mm / 2.0) * s)
        cv2.circle(escena, (cx, cy), r_ext, color_corona_bgr, -1, cv2.LINE_AA)
        if color_nucleo_bgr is not None:
            r_int = int(r_ext * 0.58)
            cv2.circle(escena, (cx, cy), r_int, color_nucleo_bgr, -1, cv2.LINE_AA)

    # 1. Cuatro billetes en las esquinas con rotaciones arbitrarias
    dibujar_billete_rotado((135, 150), (133.0, 72.0), 18.0, (195, 135, 65))   # 20 EUR (Azul)
    dibujar_billete_rotado((365, 155), (140.0, 77.0), -15.0, (75, 145, 225))  # 50 EUR (Naranja)
    dibujar_billete_rotado((135, 395), (127.0, 67.0), -14.0, (95, 95, 215))   # 10 EUR (Rojo)
    dibujar_billete_rotado((365, 395), (120.0, 62.0), 24.0, (155, 160, 155))  # 5 EUR (Gris)

    # Colores metálicos representativos en BGR
    color_plata = (185, 188, 190)
    color_oro = (65, 185, 225)
    color_cobre = (65, 105, 195)

    # 2. Fila central superior: Bimetálicas (2€, 1€) + Oro Nórdico (50c, 20c)
    dibujar_moneda((140, 260), 25.75, color_corona_bgr=color_plata, color_nucleo_bgr=color_oro) # 2€
    dibujar_moneda((215, 260), 23.25, color_corona_bgr=color_oro, color_nucleo_bgr=color_plata) # 1€
    dibujar_moneda((290, 260), 24.25, color_corona_bgr=color_oro)  # 50c
    dibujar_moneda((365, 260), 22.25, color_corona_bgr=color_oro)  # 20c

    # 3. Fila central inferior: Oro Nórdico (10c) + Familia Cobre (5c, 2c, 1c)
    dibujar_moneda((140, 318), 19.75, color_corona_bgr=color_oro)   # 10c
    dibujar_moneda((215, 318), 21.25, color_corona_bgr=color_cobre) # 5c
    dibujar_moneda((290, 318), 18.75, color_corona_bgr=color_cobre) # 2c
    dibujar_moneda((365, 318), 16.25, color_corona_bgr=color_cobre) # 1c

    return escena


# ==============================================================================
# 9. LECTURA/ESCRITURA SEGURA EN WINDOWS Y PIPELINE PRINCIPAL
# ==============================================================================
def leer_imagen_unicode(ruta: str) -> Optional[np.ndarray]:
    """
    Lee una imagen desde disco soportando rutas con tildes, eñes o caracteres
    especiales en Windows (donde cv2.imread falla).
    """
    try:
        datos = np.fromfile(ruta, dtype=np.uint8)
        if datos.size == 0:
            return None
        return cv2.imdecode(datos, cv2.IMREAD_COLOR)
    except Exception:
        return None


def guardar_imagen_unicode(ruta: str, img_bgr: np.ndarray) -> bool:
    """
    Guarda una imagen en disco soportando rutas con caracteres especiales en Windows.
    """
    try:
        ext = os.path.splitext(ruta)[1] or ".png"
        exito, buffer = cv2.imencode(ext, img_bgr)
        if not exito:
            return False
        buffer.tofile(ruta)
        return True
    except Exception:
        return False


def procesar_fotograma(
    img_bgr: np.ndarray,
    cfg: ConfiguracionSistema,
    imprimir_traza: bool = True
) -> Tuple[np.ndarray, np.ndarray, List[ResultadoClasificacion], Dict]:
    """Ejecuta el pipeline completo sobre un fotograma BGR."""
    mascara, contornos = segmentar_escena(img_bgr, cfg)

    resultados: List[ResultadoClasificacion] = []
    for idx, cnt in enumerate(contornos, start=1):
        feat = extraer_caracteristicas(img_bgr, cnt, idx, cfg)
        res = clasificar_objeto_arbol(feat, cfg)
        resultados.append(res)

        if imprimir_traza:
            camino_str = " -> ".join(res.ruta_arbol)
            print(f"[Objeto #{idx:02d}] Clase: {res.etiqueta:10s} | {camino_str}")

    img_hud, resumen = interpretar_y_visualizar(img_bgr, resultados, cfg)
    return img_hud, mascara, resultados, resumen


# ==============================================================================
# 10. INTERFAZ GRÁFICA (GUI) CON EXPLORADOR DE ARCHIVOS AL INICIAR
# ==============================================================================
class InterfazMonedasBilletes:
    """
    Interfaz gráfica en Tkinter que al iniciar abre automáticamente el explorador
    de archivos para seleccionar la imagen de trabajo y muestra el resultado del
    sistema experto, la máscara de segmentación y la traza del árbol de decisión.
    """
    def __init__(self, root: tk.Tk, cfg: ConfiguracionSistema, ruta_inicial: Optional[str] = None):
        self.root = root
        self.cfg = cfg
        self.root.title("Visión Artificial Clásica - Detector de Monedas y Billetes de Euro")
        self.root.geometry("1280x820")
        self.root.minsize(1024, 680)
        self.root.configure(bg="#1e242b")

        self.img_original: Optional[np.ndarray] = None
        self.img_hud: Optional[np.ndarray] = None
        self.img_mascara: Optional[np.ndarray] = None
        self.resultados: List[ResultadoClasificacion] = []
        self.resumen: Dict = {}
        self.ruta_actual: Optional[str] = ruta_inicial
        self.photo_actual: Optional[tk.PhotoImage] = None
        self.modo_vista = tk.StringVar(value="HUD")
        self.var_escala = tk.DoubleVar(value=round(self.cfg.px_por_mm, 3))
        self.var_ref_calibracion = tk.StringVar(
            value="Moneda 2.00 EUR (2 euros - 25.75 mm)"
        )
        self.modo_clic_calibrar: bool = False
        self._transform_lienzo: Tuple[int, int, float] = (0, 0, 1.0)  # (offset_x, offset_y, escala)

        self._construir_ui()

        # Al iniciar la ventana, abrir inmediatamente el explorador de archivos
        # (a menos que ya se haya pasado una ruta por línea de comandos)
        if self.ruta_actual:
            self.root.after(100, lambda: self.cargar_desde_ruta(self.ruta_actual))
        else:
            self.root.after(150, self.seleccionar_imagen_explorador)

    def _construir_ui(self):
        # --- Barra superior de herramientas ---
        barra = tk.Frame(self.root, bg="#14181d", pady=10, padx=14)
        barra.pack(side=tk.TOP, fill=tk.X)

        btn_estilo = {
            "font": ("Segoe UI", 10, "bold"),
            "relief": tk.FLAT,
            "padx": 12,
            "pady": 6,
            "cursor": "hand2"
        }

        tk.Button(
            barra, text="📂 Seleccionar Imagen...",
            bg="#00b894", fg="#ffffff", activebackground="#00a884",
            command=self.seleccionar_imagen_explorador, **btn_estilo
        ).pack(side=tk.LEFT, padx=(0, 8))

        tk.Button(
            barra, text="🧪 Cargar Escena Sintética",
            bg="#0984e3", fg="#ffffff", activebackground="#0773c5",
            command=self.cargar_escena_sintetica, **btn_estilo
        ).pack(side=tk.LEFT, padx=4)

        tk.Button(
            barra, text="📏 Calibrar con Moneda / Billete...",
            bg="#6c5ce7", fg="#ffffff", activebackground="#5b4bc4",
            command=self.abrir_dialogo_calibracion, **btn_estilo
        ).pack(side=tk.LEFT, padx=4)

        self.btn_clic_cal = tk.Button(
            barra, text="🎯 Clic en Objeto para Calibrar",
            bg="#a29bfe", fg="#14181d", activebackground="#8c7ae6",
            command=self.activar_modo_clic_calibracion, **btn_estilo
        )
        self.btn_clic_cal.pack(side=tk.LEFT, padx=4)

        tk.Button(
            barra, text="💾 Guardar Resultado...",
            bg="#fdcb6e", fg="#1e242b", activebackground="#e5b75f",
            command=self.guardar_resultado, **btn_estilo
        ).pack(side=tk.LEFT, padx=4)

        # Control manual de escala px/mm
        frame_escala = tk.Frame(barra, bg="#14181d")
        frame_escala.pack(side=tk.RIGHT, padx=6)

        tk.Label(
            frame_escala, text="Escala (px/mm):",
            bg="#14181d", fg="#dfe6e9", font=("Segoe UI", 9, "bold")
        ).pack(side=tk.LEFT, padx=(0, 4))

        spin_escala = tk.Spinbox(
            frame_escala, from_=0.30, to=20.00, increment=0.05,
            textvariable=self.var_escala, width=7, font=("Segoe UI", 10),
            command=self.aplicar_escala_manual
        )
        spin_escala.pack(side=tk.LEFT, padx=(0, 4))
        spin_escala.bind("<Return>", lambda _: self.aplicar_escala_manual())

        tk.Button(
            frame_escala, text="Aplicar",
            bg="#2d3436", fg="#ffffff", relief=tk.FLAT, padx=8, pady=3,
            font=("Segoe UI", 9), command=self.aplicar_escala_manual
        ).pack(side=tk.LEFT)

        # --- Selector de Modo de Vista ---
        barra_vista = tk.Frame(self.root, bg="#252c35", pady=6, padx=14)
        barra_vista.pack(side=tk.TOP, fill=tk.X)

        tk.Label(
            barra_vista, text="Vista activa:",
            bg="#252c35", fg="#b2bec3", font=("Segoe UI", 9, "bold")
        ).pack(side=tk.LEFT, padx=(0, 10))

        for texto, valor in [
            ("Resultado Clasificado (HUD)", "HUD"),
            ("Máscara Segmentada (HSV + Morfología)", "MASCARA"),
            ("Imagen Original", "ORIGINAL")
        ]:
            tk.Radiobutton(
                barra_vista, text=texto, variable=self.modo_vista, value=valor,
                bg="#252c35", fg="#ffffff", selectcolor="#14181d",
                activebackground="#252c35", activeforeground="#00cec9",
                font=("Segoe UI", 9), command=self._actualizar_lienzo
            ).pack(side=tk.LEFT, padx=8)

        self.lbl_archivo = tk.Label(
            barra_vista, text="Ningún archivo seleccionado",
            bg="#252c35", fg="#81ecec", font=("Segoe UI", 9, "italic")
        )
        self.lbl_archivo.pack(side=tk.RIGHT)

        # --- Panel central dividido: Lienzo de Imagen (Izquierda) + Informe (Derecha) ---
        panel_central = tk.PanedWindow(
            self.root, orient=tk.HORIZONTAL, bg="#1e242b",
            sashwidth=6, sashrelief=tk.RAISED
        )
        panel_central.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Lienzo de visualización de imagen
        frame_img = tk.Frame(panel_central, bg="#111519", bd=1, relief=tk.SUNKEN)
        panel_central.add(frame_img, minsize=580, stretch="always")

        self.canvas_img = tk.Canvas(frame_img, bg="#111519", highlightthickness=0)
        self.canvas_img.pack(fill=tk.BOTH, expand=True)
        self.canvas_img.bind("<Configure>", lambda _: self._actualizar_lienzo())
        self.canvas_img.bind("<Button-1>", self._al_hacer_clic_lienzo)

        # Panel lateral derecho: Resumen contable y traza del árbol de decisión
        frame_info = tk.Frame(panel_central, bg="#181d23", padx=12, pady=12)
        panel_central.add(frame_info, minsize=360, stretch="never")

        tk.Label(
            frame_info, text="RESUMEN DE LA ESCENA",
            bg="#181d23", fg="#00cec9", font=("Segoe UI", 11, "bold")
        ).pack(anchor="w", pady=(0, 6))

        self.lbl_importe = tk.Label(
            frame_info, text="IMPORTE TOTAL: 0.00 EUR",
            bg="#111519", fg="#00ff8c", font=("Segoe UI", 16, "bold"),
            padx=12, pady=10, anchor="w"
        )
        self.lbl_importe.pack(fill=tk.X, pady=(0, 8))

        self.lbl_conteo = tk.Label(
            frame_info, text="Monedas: 0 | Billetes: 0 | Alertas: 0",
            bg="#181d23", fg="#dfe6e9", font=("Segoe UI", 10), anchor="w"
        )
        self.lbl_conteo.pack(fill=tk.X, pady=(0, 10))

        tk.Label(
            frame_info, text="TRAZA DEL ÁRBOL DE DECISIÓN E INVENTARIO",
            bg="#181d23", fg="#00cec9", font=("Segoe UI", 10, "bold")
        ).pack(anchor="w", pady=(4, 4))

        frame_texto = tk.Frame(frame_info, bg="#111519")
        frame_texto.pack(fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(frame_texto, orient=tk.VERTICAL)
        self.txt_traza = tk.Text(
            frame_texto, bg="#111519", fg="#dfe6e9",
            font=("Consolas", 9), wrap=tk.WORD,
            yscrollcommand=scrollbar.set, relief=tk.FLAT, padx=8, pady=8
        )
        scrollbar.config(command=self.txt_traza.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.txt_traza.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    def seleccionar_imagen_explorador(self):
        """Abre el explorador de archivos nativo para elegir una imagen."""
        ruta = filedialog.askopenfilename(
            parent=self.root,
            title="Seleccionar imagen cenital de monedas y billetes",
            initialdir=os.getcwd(),
            filetypes=[
                ("Archivos de imagen", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp"),
                ("Todos los archivos", "*.*")
            ]
        )
        if ruta:
            self.cargar_desde_ruta(ruta)
        elif self.img_original is None:
            # Si el usuario cierra el explorador sin elegir imagen al inicio,
            # mostramos un mensaje de ayuda en el lienzo
            self._mostrar_mensaje_bienvenida()

    def cargar_desde_ruta(self, ruta: str):
        """Lee la imagen seleccionada en el explorador y ejecuta el análisis."""
        img = leer_imagen_unicode(ruta)
        if img is None:
            messagebox.showerror(
                "Error al abrir imagen",
                f"No se pudo cargar el archivo seleccionado:\n{ruta}"
            )
            return

        self.ruta_actual = ruta
        nombre = os.path.basename(ruta)
        self.lbl_archivo.config(text=f"Archivo: {nombre} ({img.shape[1]}x{img.shape[0]} px)")
        self.img_original = img
        self._ejecutar_analisis()

    def cargar_escena_sintetica(self):
        """Genera y analiza la escena sintética de referencia."""
        self.ruta_actual = None
        self.img_original = generar_escena_sintetica_prueba(self.cfg)
        self.lbl_archivo.config(
            text=f"Escena sintética de prueba ({self.img_original.shape[1]}x{self.img_original.shape[0]} px)"
        )
        self._ejecutar_analisis()

    def aplicar_escala_manual(self):
        """Actualiza el factor px/mm y recalcula la clasificación."""
        try:
            nueva_escala = float(self.var_escala.get())
            if nueva_escala <= 0:
                raise ValueError
            self.cfg.px_por_mm = nueva_escala
            if self.img_original is not None:
                self._ejecutar_analisis()
        except ValueError:
            messagebox.showwarning("Escala inválida", "Introduce un valor numérico positivo para px/mm.")

    def abrir_dialogo_calibracion(self):
        """
        Abre una ventana emergente que permite elegir cualquier moneda (1c a 2€)
        o billete (5€ a 50€) como referencia de calibración, así como qué objeto
        detectado en la imagen corresponde a dicha denominación.
        """
        if self.img_original is None:
            messagebox.showinfo("Sin imagen", "Primero selecciona una imagen para calibrar la escala.")
            return

        dlg = tk.Toplevel(self.root)
        dlg.title("Calibrar Escala con Moneda o Billete")
        dlg.geometry("520x390")
        dlg.resizable(False, False)
        dlg.configure(bg="#181d23", padx=18, pady=16)
        dlg.transient(self.root)
        dlg.grab_set()

        tk.Label(
            dlg, text="CALIBRACIÓN DE ESCALA ESPACIAL (px/mm)",
            bg="#181d23", fg="#00cec9", font=("Segoe UI", 12, "bold")
        ).pack(anchor="w", pady=(0, 6))

        tk.Label(
            dlg,
            text=(
                "Elige la moneda o el billete real que vas a usar como patrón de medida\n"
                "y selecciona a qué objeto de la imagen corresponde:"
            ),
            bg="#181d23", fg="#dfe6e9", font=("Segoe UI", 9), justify=tk.LEFT
        ).pack(anchor="w", pady=(0, 12))

        # 1. Selector de Denominación (cualquier moneda o billete)
        tk.Label(
            dlg, text="1. Moneda o billete de referencia:",
            bg="#181d23", fg="#81ecec", font=("Segoe UI", 10, "bold")
        ).pack(anchor="w", pady=(0, 4))

        opciones_ref = list(REFERENCIAS_CALIBRACION.keys())
        combo_ref = ttk.Combobox(
            dlg, textvariable=self.var_ref_calibracion,
            values=opciones_ref, state="readonly", font=("Segoe UI", 10), width=48
        )
        combo_ref.pack(fill=tk.X, pady=(0, 14))

        # 2. Selector de Objeto dentro de la escena
        tk.Label(
            dlg, text="2. Objeto de la escena al que corresponde:",
            bg="#181d23", fg="#81ecec", font=("Segoe UI", 10, "bold")
        ).pack(anchor="w", pady=(0, 4))

        mapa_objetos: Dict[str, Optional[np.ndarray]] = {
            "Automático: Objeto compatible de MAYOR tamaño en la imagen": None,
            "Automático: Objeto compatible de MENOR tamaño en la imagen": None,
        }

        # Añadir todos los objetos actualmente numerados en el HUD (#1, #2, ...)
        if self.resultados:
            for res in self.resultados:
                f = res.caracteristicas
                forma = "Circular" if f.circularidad >= 0.72 else "Rectangular"
                clave_obj = (
                    f"Objeto #{res.id_objeto:02d} ({forma} - actual: {res.etiqueta} | "
                    f"centro: {f.centroide_px[0]},{f.centroide_px[1]} px)"
                )
                mapa_objetos[clave_obj] = f.contorno
        else:
            # Si aún no había objetos dentro del rango de área actual, listar candidatos brutos
            candidatos = obtener_contornos_candidatos_calibracion(self.img_original, self.cfg)
            for idx, cnt in enumerate(candidatos, start=1):
                area_px = cv2.contourArea(cnt)
                perim_px = cv2.arcLength(cnt, True)
                circ = (4.0 * np.pi * area_px) / (perim_px ** 2 + 1e-6)
                forma = "Circular" if circ >= 0.72 else "Rectangular"
                clave_obj = f"Candidato #{idx:02d} ({forma} - área: {area_px:.0f} px²)"
                mapa_objetos[clave_obj] = cnt

        lista_claves_obj = list(mapa_objetos.keys())
        var_obj_elegido = tk.StringVar(value=lista_claves_obj[0])
        combo_obj = ttk.Combobox(
            dlg, textvariable=var_obj_elegido,
            values=lista_claves_obj, state="readonly", font=("Segoe UI", 9), width=48
        )
        combo_obj.pack(fill=tk.X, pady=(0, 14))

        tk.Label(
            dlg,
            text=(
                "💡 Consejo: También puedes pulsar '🎯 Elegir haciendo clic en la imagen'\n"
                "y pinchar directamente sobre cualquier moneda o billete con el ratón."
            ),
            bg="#111519", fg="#fdcb6e", font=("Segoe UI", 8, "italic"),
            padx=10, pady=8, justify=tk.LEFT
        ).pack(fill=tk.X, pady=(0, 16))

        frame_btns = tk.Frame(dlg, bg="#181d23")
        frame_btns.pack(fill=tk.X)

        def confirmar_calibracion():
            clave_ref = self.var_ref_calibracion.get()
            clave_obj = var_obj_elegido.get()
            contorno_esp = mapa_objetos.get(clave_obj)
            criterio = "MENOR" if "MENOR" in clave_obj else "MAYOR"
            try:
                nueva_escala = autocalibrar_con_referencia(
                    self.img_original, self.cfg, clave_ref,
                    contorno_especifico=contorno_esp, criterio_auto=criterio
                )
                self.var_escala.set(round(nueva_escala, 3))
                self._ejecutar_analisis()
                dlg.destroy()
                messagebox.showinfo(
                    "Calibración completada",
                    f"Referencia aplicada: {clave_ref}\n"
                    f"Nueva escala calculada: {nueva_escala:.3f} px/mm"
                )
            except ValueError as e:
                messagebox.showwarning("No se pudo calibrar", str(e), parent=dlg)

        def pasar_a_modo_clic():
            dlg.destroy()
            self.activar_modo_clic_calibracion()

        tk.Button(
            frame_btns, text="✅ Calibrar Ahora",
            bg="#00b894", fg="#ffffff", activebackground="#00a884",
            font=("Segoe UI", 10, "bold"), relief=tk.FLAT, padx=14, pady=6,
            cursor="hand2", command=confirmar_calibracion
        ).pack(side=tk.LEFT, padx=(0, 8))

        tk.Button(
            frame_btns, text="🎯 Elegir haciendo clic en la imagen",
            bg="#6c5ce7", fg="#ffffff", activebackground="#5b4bc4",
            font=("Segoe UI", 9, "bold"), relief=tk.FLAT, padx=10, pady=6,
            cursor="hand2", command=pasar_a_modo_clic
        ).pack(side=tk.LEFT, padx=4)

        tk.Button(
            frame_btns, text="Cancelar",
            bg="#2d3436", fg="#dfe6e9", relief=tk.FLAT, padx=10, pady=6,
            font=("Segoe UI", 9), command=dlg.destroy
        ).pack(side=tk.RIGHT)

    def activar_modo_clic_calibracion(self):
        """Activa el modo interactivo para calibrar pinchando sobre una moneda o billete."""
        if self.img_original is None:
            messagebox.showinfo("Sin imagen", "Primero selecciona una imagen para calibrar.")
            return
        self.modo_clic_calibrar = not self.modo_clic_calibrar
        if self.modo_clic_calibrar:
            self.btn_clic_cal.config(
                text="🎯 Haz clic en una moneda o billete... (Cancelar)",
                bg="#e17055", fg="#ffffff"
            )
            self.canvas_img.config(cursor="crosshair")
        else:
            self.btn_clic_cal.config(
                text="🎯 Clic en Objeto para Calibrar",
                bg="#a29bfe", fg="#14181d"
            )
            self.canvas_img.config(cursor="")

    def _al_hacer_clic_lienzo(self, event):
        """Detecta si el usuario ha hecho clic sobre una moneda o billete para calibrar con él."""
        if not self.modo_clic_calibrar or self.img_original is None:
            return

        off_x, off_y, escala = self._transform_lienzo
        if escala <= 0:
            return

        # Convertir coordenadas del clic en el canvas a píxeles reales de la imagen original
        x_img = int((event.x - off_x) / escala)
        y_img = int((event.y - off_y) / escala)
        h_img, w_img = self.img_original.shape[:2]
        if not (0 <= x_img < w_img and 0 <= y_img < h_img):
            return

        # Buscar qué contorno contiene el punto pinchado (o el más cercano)
        candidatos = (
            [r.caracteristicas.contorno for r in self.resultados]
            if self.resultados else
            obtener_contornos_candidatos_calibracion(self.img_original, self.cfg)
        )
        if not candidatos:
            messagebox.showwarning("Sin objetos", "No se detectaron objetos en la escena.")
            return

        contorno_seleccionado = None
        mejor_dist = float("inf")
        for cnt in candidatos:
            dist = cv2.pointPolygonTest(cnt, (float(x_img), float(y_img)), True)
            if dist >= 0:
                contorno_seleccionado = cnt
                break
            if abs(dist) < mejor_dist:
                mejor_dist = abs(dist)
                if mejor_dist <= 45.0:
                    contorno_seleccionado = cnt

        if contorno_seleccionado is None:
            messagebox.showinfo(
                "Fuera de objeto",
                "Haz clic directamente encima de una moneda o de un billete."
            )
            return

        # Salir del modo clic y preguntar qué moneda o billete es el objeto pinchado
        self.activar_modo_clic_calibracion()
        self._abrir_popup_calibrar_contorno(contorno_seleccionado)

    def _abrir_popup_calibrar_contorno(self, contorno: np.ndarray):
        """Pregunta al usuario qué moneda o billete es el objeto sobre el que acaba de hacer clic."""
        area_px = float(cv2.contourArea(contorno))
        perim_px = float(cv2.arcLength(contorno, True))
        circ = (4.0 * np.pi * area_px) / (perim_px ** 2 + 1e-6)
        es_circular = circ >= 0.72

        # Preseleccionar por defecto 2€ si parece circular o 20€ si parece rectangular
        if es_circular and "Billete" in self.var_ref_calibracion.get():
            self.var_ref_calibracion.set("Moneda 2.00 EUR (2 euros - 25.75 mm)")
        elif not es_circular and "Moneda" in self.var_ref_calibracion.get():
            self.var_ref_calibracion.set("Billete 20.00 EUR (20 euros - 133x72 mm)")

        dlg = tk.Toplevel(self.root)
        dlg.title("¿Qué moneda o billete has seleccionado?")
        dlg.geometry("460x230")
        dlg.resizable(False, False)
        dlg.configure(bg="#181d23", padx=18, pady=16)
        dlg.transient(self.root)
        dlg.grab_set()

        tipo_detectado = "Moneda circular" if es_circular else "Billete rectangular"
        tk.Label(
            dlg,
            text=f"Objeto seleccionado ({tipo_detectado}, área: {area_px:.0f} px²)",
            bg="#181d23", fg="#00cec9", font=("Segoe UI", 10, "bold")
        ).pack(anchor="w", pady=(0, 8))

        tk.Label(
            dlg, text="Indica qué moneda o billete real es para calibrar la escala:",
            bg="#181d23", fg="#dfe6e9", font=("Segoe UI", 9)
        ).pack(anchor="w", pady=(0, 6))

        combo = ttk.Combobox(
            dlg, textvariable=self.var_ref_calibracion,
            values=list(REFERENCIAS_CALIBRACION.keys()),
            state="readonly", font=("Segoe UI", 10), width=44
        )
        combo.pack(fill=tk.X, pady=(0, 18))

        frame_b = tk.Frame(dlg, bg="#181d23")
        frame_b.pack(fill=tk.X)

        def aplicar():
            clave_ref = self.var_ref_calibracion.get()
            nueva_escala = autocalibrar_con_referencia(
                self.img_original, self.cfg, clave_ref, contorno_especifico=contorno
            )
            self.var_escala.set(round(nueva_escala, 3))
            self._ejecutar_analisis()
            dlg.destroy()
            messagebox.showinfo(
                "Calibración completada",
                f"Calibrado con: {clave_ref}\nNueva escala: {nueva_escala:.3f} px/mm"
            )

        tk.Button(
            frame_b, text="✅ Calibrar con este objeto",
            bg="#00b894", fg="#ffffff", font=("Segoe UI", 10, "bold"),
            relief=tk.FLAT, padx=14, pady=6, cursor="hand2", command=aplicar
        ).pack(side=tk.LEFT)

        tk.Button(
            frame_b, text="Cancelar",
            bg="#2d3436", fg="#dfe6e9", font=("Segoe UI", 9),
            relief=tk.FLAT, padx=10, pady=6, command=dlg.destroy
        ).pack(side=tk.RIGHT)

    def guardar_resultado(self):
        """Abre un diálogo para guardar la imagen procesada."""
        if self.img_hud is None:
            messagebox.showinfo("Sin resultado", "No hay ninguna imagen procesada para guardar.")
            return

        ruta_destino = filedialog.asksaveasfilename(
            parent=self.root,
            title="Guardar imagen procesada con HUD",
            defaultextension=".png",
            initialfile="resultado_monedas_billetes.png",
            filetypes=[
                ("Imagen PNG", "*.png"),
                ("Imagen JPEG", "*.jpg"),
                ("Todos los archivos", "*.*")
            ]
        )
        if ruta_destino:
            if guardar_imagen_unicode(ruta_destino, self.img_hud):
                messagebox.showinfo("Guardado con éxito", f"Imagen guardada en:\n{ruta_destino}")
            else:
                messagebox.showerror("Error", "No se pudo guardar la imagen en la ruta indicada.")

    def _ejecutar_analisis(self):
        """Procesa `self.img_original` y actualiza tanto el lienzo como el informe lateral."""
        if self.img_original is None:
            return

        print("-" * 80)
        self.img_hud, self.img_mascara, self.resultados, self.resumen = procesar_fotograma(
            self.img_original, self.cfg, imprimir_traza=True
        )
        print("-" * 80)

        # Actualizar indicadores del panel derecho
        self.lbl_importe.config(text=f"IMPORTE TOTAL: {self.resumen['total_euros']:.2f} EUR")
        self.lbl_conteo.config(
            text=(
                f"Monedas: {self.resumen['num_monedas']}  |  "
                f"Billetes: {self.resumen['num_billetes']}  |  "
                f"Alertas/Intrusos: {self.resumen['num_alertas']}"
            )
        )

        # Construir el texto de desglose y traza por objeto
        lineas = []
        lineas.append("=== DESGLOSE POR DENOMINACIÓN ===")
        if self.resumen["desglose"]:
            for etq, cant in sorted(self.resumen["desglose"].items()):
                lineas.append(f"  • {etq:10s} : {cant} ud(s)")
        else:
            lineas.append("  (Sin objetos válidos clasificados)")

        lineas.append("\n=== TRAZA DEL ÁRBOL DE DECISIÓN ===")
        for res in self.resultados:
            f = res.caracteristicas
            lineas.append(
                f"\n[Objeto #{res.id_objeto:02d}] -> {res.etiqueta} ({res.categoria})"
            )
            lineas.append(
                f"  Área: {f.area_mm2:.1f} mm² | Lados: {f.lado_mayor_mm:.1f}x{f.lado_menor_mm:.1f} mm"
            )
            lineas.append(
                f"  Circ: {f.circularidad:.2f} | AR: {f.relacion_aspecto:.2f} | Δb*: {f.delta_b_radial:+.1f}"
            )
            for paso in res.ruta_arbol:
                lineas.append(f"   ↳ {paso}")

        self.txt_traza.config(state=tk.NORMAL)
        self.txt_traza.delete("1.0", tk.END)
        self.txt_traza.insert(tk.END, "\n".join(lineas))
        self.txt_traza.config(state=tk.DISABLED)

        self._actualizar_lienzo()

    def _mostrar_mensaje_bienvenida(self):
        self.canvas_img.delete("all")
        w = max(self.canvas_img.winfo_width(), 400)
        h = max(self.canvas_img.winfo_height(), 300)
        self.canvas_img.create_text(
            w // 2, h // 2,
            text=(
                "Haz clic en '📂 Seleccionar Imagen...' arriba a la izquierda\n"
                "para abrir el explorador de archivos y elegir una imagen,\n"
                "o pulsa '🧪 Cargar Escena Sintética' para probar el sistema."
            ),
            fill="#b2bec3", font=("Segoe UI", 12), justify=tk.CENTER
        )

    def _actualizar_lienzo(self):
        """Redimensiona la imagen activa ajustándola al lienzo sin deformar su aspecto."""
        if self.img_original is None:
            self._mostrar_mensaje_bienvenida()
            return

        modo = self.modo_vista.get()
        if modo == "HUD" and self.img_hud is not None:
            img_mostrar = self.img_hud
        elif modo == "MASCARA" and self.img_mascara is not None:
            img_mostrar = cv2.cvtColor(self.img_mascara, cv2.COLOR_GRAY2BGR)
        else:
            img_mostrar = self.img_original

        canvas_w = self.canvas_img.winfo_width()
        canvas_h = self.canvas_img.winfo_height()
        if canvas_w < 20 or canvas_h < 20:
            return

        h_img, w_img = img_mostrar.shape[:2]
        escala = min(canvas_w / w_img, canvas_h / h_img)
        nuevo_w = max(1, int(w_img * escala))
        nuevo_h = max(1, int(h_img * escala))
        off_x = (canvas_w - nuevo_w) // 2
        off_y = (canvas_h - nuevo_h) // 2
        self._transform_lienzo = (off_x, off_y, escala)

        interp = cv2.INTER_AREA if escala < 1.0 else cv2.INTER_LINEAR
        img_redim = cv2.resize(img_mostrar, (nuevo_w, nuevo_h), interpolation=interp)

        # Codificar en PNG en memoria para PhotoImage de Tkinter (sin requerir PIL/Pillow)
        exito, buffer = cv2.imencode(".png", img_redim)
        if not exito:
            return
        b64_data = base64.b64encode(buffer)
        self.photo_actual = tk.PhotoImage(data=b64_data)

        self.canvas_img.delete("all")
        self.canvas_img.create_image(
            canvas_w // 2, canvas_h // 2,
            image=self.photo_actual, anchor=tk.CENTER
        )


def lanzar_interfaz_grafica(cfg: ConfiguracionSistema, ruta_inicial: Optional[str] = None):
    """Inicializa y ejecuta la ventana principal de la aplicación."""
    root = tk.Tk()
    InterfazMonedasBilletes(root, cfg, ruta_inicial=ruta_inicial)
    root.mainloop()


def main():
    parser = argparse.ArgumentParser(
        description="Sistema Clásico de Visión Artificial: Monedas y Billetes"
    )
    parser.add_argument("--imagen", type=str, default=None, help="Ruta a imagen cenital de entrada")
    parser.add_argument("--camara", type=int, default=None, help="Índice de la cámara USB/cenital (ej. 0)")
    parser.add_argument("--px_mm", type=float, default=2.16, help="Factor de escala en píxeles por milímetro")
    parser.add_argument("--guardar", type=str, default="resultado_monedas_billetes.png", help="Ruta para guardar el resultado")
    parser.add_argument("--sin-gui", action="store_true", help="Ejecutar solo por consola sin abrir la interfaz gráfica")
    args = parser.parse_args()

    cfg = ConfiguracionSistema(px_por_mm=args.px_mm)

    if args.camara is not None:
        cap = cv2.VideoCapture(args.camara)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        cap.set(cv2.CAP_PROP_AUTO_WB, 0)
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)

        print("Evaluación en tiempo real iniciada:")
        print("  [q] Salir | [s] Guardar captura | [c] Autocalibrar escala con moneda de 2 EUR")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            vista_hud, mascara, _, _ = procesar_fotograma(frame, cfg, imprimir_traza=False)
            cv2.imshow("Vision Clasica - Monedas y Billetes (HUD)", vista_hud)
            cv2.imshow("Mascara Segmentada (HSV + Morfologia)", mascara)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                guardar_imagen_unicode(args.guardar, vista_hud)
                print(f"Captura guardada en {args.guardar}")
            elif key == ord("c"):
                try:
                    nueva_escala = autocalibrar_por_moneda_mayor(frame, cfg, 25.75)
                    print(f"[Calibración OK] Nueva escala fijada en: {nueva_escala:.3f} px/mm")
                except ValueError as e:
                    print(f"[Error Calibración] {e}")

        cap.release()
        cv2.destroyAllWindows()

    elif not args.sin_gui:
        # Por defecto, abre la interfaz gráfica y lanza inmediatamente el explorador de archivos
        lanzar_interfaz_grafica(cfg, ruta_inicial=args.imagen)

    else:
        if args.imagen is not None:
            img_entrada = leer_imagen_unicode(args.imagen)
            if img_entrada is None:
                raise FileNotFoundError(f"No se pudo abrir la imagen: {args.imagen}")
        else:
            print("Generando escena sintética de validación con las 8 monedas y 4 billetes...")
            img_entrada = generar_escena_sintetica_prueba(cfg)

        print("-" * 80)
        vista_hud, _, _, resumen = procesar_fotograma(img_entrada, cfg, imprimir_traza=True)
        print("-" * 80)
        print(f"RESUMEN FINAL -> Importe Total: {resumen['total_euros']:.2f} EUR "
              f"({resumen['num_monedas']} monedas, {resumen['num_billetes']} billetes)")
        guardar_imagen_unicode(args.guardar, vista_hud)
        print(f"Imagen de salida con HUD guardada en: {args.guardar}")


if __name__ == "__main__":
    main()
