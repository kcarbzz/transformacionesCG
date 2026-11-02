# SC3321 Vision Computacional : 584431 / 696641 : Xavier Maximiliano López Ibarra / Ronal Waldo Mora Aybar
"""Escalamiento, skew, rotación y traslación de una imagen.

Usage:
  transformaciones.py <entrada> [--salida=<archivo>] [--escala=<sx,sy>]
                      [--skew=<kx,ky>] [--rotacion=<grados>]
                      [--traslacion=<tx,ty>]
  transformaciones.py -h | --help

Options:
  -h --help               Muestra esta ayuda.
  --salida=<archivo>      Imagen de salida (PNG) [default: salida.png].
  --escala=<sx,sy>        Factores de escalamiento.
  --skew=<kx,ky>          Inclinación: x' = x + kx*y, y' = y + ky*x.
  --rotacion=<grados>     Rotación antihoraria alrededor del centro.
  --traslacion=<tx,ty>    Desplazamiento en píxeles (x a la derecha, y arriba).

Las transformaciones indicadas se combinan y se aplican en el orden:
escala -> skew -> rotación -> traslación. Escala, skew y rotación se hacen
respecto al centro de la imagen.
"""

import cv2
import numpy as np
from docopt import docopt


def escalamiento(sx, sy):
    return np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]], dtype=float)


def skew(kx, ky):
    return np.array([[1, kx, 0], [ky, 1, 0], [0, 0, 1]], dtype=float)


def rotacion(grados):
    a = np.radians(grados)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def traslacion(tx, ty):
    return np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=float)


def aplicar(T, puntos):
    """Aplica la matriz 3x3 T a una matriz de puntos 2 x N."""
    homogeneos = np.vstack([puntos, np.ones(puntos.shape[1])])
    return (T @ homogeneos)[:2]


def transformar_imagen(imagen, T):
    """Devuelve la imagen transformada por T.

    Las coordenadas tienen el origen en el centro de la imagen y el eje y
    hacia arriba. Para cada píxel de la salida se busca de qué píxel de la
    entrada viene usando la inversa de T, así no quedan huecos.
    """
    alto, ancho = imagen.shape[:2]
    cx, cy = (ancho - 1) / 2, (alto - 1) / 2

    # El lienzo de salida se ajusta a las esquinas transformadas
    esquinas = np.array([[-cx, cx, cx, -cx], [cy, cy, -cy, -cy]])
    nuevas = aplicar(T, esquinas)
    xmin, ymin = nuevas.min(axis=1)
    xmax, ymax = nuevas.max(axis=1)
    ancho_s = int(np.ceil(xmax - xmin)) + 1
    alto_s = int(np.ceil(ymax - ymin)) + 1

    columnas, filas = np.meshgrid(np.arange(ancho_s), np.arange(alto_s))
    destino = np.vstack([(columnas + xmin).ravel(), (ymax - filas).ravel()])
    origen = aplicar(np.linalg.inv(T), destino)

    col = np.rint(origen[0] + cx).astype(int)
    fil = np.rint(cy - origen[1]).astype(int)
    dentro = (col >= 0) & (col < ancho) & (fil >= 0) & (fil < alto)

    salida = np.zeros((alto_s * ancho_s, imagen.shape[2]), dtype=imagen.dtype)
    salida[dentro] = imagen[fil[dentro], col[dentro]]
    return salida.reshape(alto_s, ancho_s, imagen.shape[2])


def par(texto):
    return [float(v) for v in texto.split(",")]


def main():
    args = docopt(__doc__)
    imagen = cv2.imread(args["<entrada>"], cv2.IMREAD_COLOR)
    if imagen is None:
        raise SystemExit(f"No se pudo abrir la imagen: {args['<entrada>']}")
    imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2BGRA)

    T = np.eye(3)
    if args["--escala"]:
        T = escalamiento(*par(args["--escala"])) @ T
    if args["--skew"]:
        T = skew(*par(args["--skew"])) @ T
    if args["--rotacion"]:
        T = rotacion(float(args["--rotacion"])) @ T
    if args["--traslacion"]:
        T = traslacion(*par(args["--traslacion"])) @ T

    np.set_printoptions(precision=3, suppress=True)
    print("Matriz de transformación:\n", T)

    resultado = transformar_imagen(imagen, T)
    cv2.imwrite(args["--salida"], resultado)
    print(f"\nImagen guardada en {args['--salida']} {resultado.shape[1]}x{resultado.shape[0]}")


if __name__ == "__main__":
    main()
