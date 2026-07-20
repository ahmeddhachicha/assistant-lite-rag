"""Chargement du corpus et découpage en segments."""


def load_corpus(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


def split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Découpe le texte en segments d'environ `chunk_size` caractères.

    Le découpage respecte d'abord les paragraphes ; les paragraphes trop
    longs sont redécoupés avec un chevauchement de `chunk_overlap`
    caractères pour ne pas couper le contexte.
    """
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]

    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        if len(para) > chunk_size:
            if current:
                chunks.append(current)
                current = ""
            start = 0
            while start < len(para):
                chunks.append(para[start : start + chunk_size])
                start += chunk_size - chunk_overlap
        elif len(current) + len(para) + 1 <= chunk_size:
            current = f"{current}\n{para}".strip()
        else:
            chunks.append(current)
            current = para
    if current:
        chunks.append(current)
    return chunks


def load_and_split(path: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    return split_text(load_corpus(path), chunk_size, chunk_overlap)
