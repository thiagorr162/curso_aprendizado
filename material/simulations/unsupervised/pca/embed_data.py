"""Injeta o mnist_embeddings.json dentro do HTML, gerando um arquivo unico.

Uso:
    python generate_mnist_embeddings.py   # cria mnist_embeddings.json
    python embed_data.py                  # injeta no HTML
"""

from pathlib import Path

HTML = Path(__file__).resolve().parent / "mnist_projections.html"
JSON = Path("mnist_embeddings.json")

START = '<script id="mnist-data" type="application/json">'
END = "</script>"


def main():
    if not JSON.exists():
        raise SystemExit(f"Nao achei {JSON}. Rode generate_mnist_embeddings.py antes.")
    if not HTML.exists():
        raise SystemExit(f"Nao achei {HTML}.")

    data = JSON.read_text(encoding="utf-8").strip()
    html = HTML.read_text(encoding="utf-8")

    i = html.index(START) + len(START)
    j = html.index(END, i)
    new_html = html[:i] + "\n" + data + "\n" + html[j:]

    HTML.write_text(new_html, encoding="utf-8")
    kb = len(data) / 1024
    print(f"Dados reais embutidos em {HTML.name} ({kb:.0f} KB de JSON).")
    print("Abra o HTML com duplo-clique: ja funciona offline, sem servidor.")


if __name__ == "__main__":
    main()
