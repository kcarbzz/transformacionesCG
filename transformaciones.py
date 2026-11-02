# SC3321 Vision Computacional : 584431 / 696641 : Xavier Maximiliano López Ibarra / Ronal Waldo Mora Aybar
"""Escalamiento, rotación y traslación de una matriz de puntos 2D usando numpy.

Usage:
  transformaciones.py [--puntos=<p>] [--escala=<sx,sy>] [--rotacion=<grados>]
                      [--traslacion=<tx,ty>]
  transformaciones.py -h | --help

Options:
  -h --help               Muestra esta ayuda.
  --puntos=<p>            Puntos "x,y;x,y;..." [default: 0,0;1,0;1,1;0,1].
  --escala=<sx,sy>        Factores de escalamiento.
  --rotacion=<grados>     Rotación antihoraria alrededor del origen.
  --traslacion=<tx,ty>    Desplazamiento en x e y.

Las transformaciones indicadas se combinan y se aplican en el orden:
escala -> rotación -> traslación.
"""

import numpy as np
from docopt import docopt


def escalamiento(sx, sy):
    return np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]], dtype=float)


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


def par(texto):
    return [float(v) for v in texto.split(",")]


def main():
    args = docopt(__doc__)
    puntos = np.array([par(p) for p in args["--puntos"].split(";")]).T

    T = np.eye(3)
    if args["--escala"]:
        T = escalamiento(*par(args["--escala"])) @ T
    if args["--rotacion"]:
        T = rotacion(float(args["--rotacion"])) @ T
    if args["--traslacion"]:
        T = traslacion(*par(args["--traslacion"])) @ T

    np.set_printoptions(precision=3, suppress=True)
    print("Matriz original:\n", puntos)
    print("\nMatriz de transformación:\n", T)
    print("\nMatriz resultante:\n", aplicar(T, puntos))


if __name__ == "__main__":
    main()
