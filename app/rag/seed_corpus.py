"""Built-in seed corpus of short, self-authored science notes.

These are original summaries written for this project (no third-party text), so
they can be freely ingested. They give the retrieval pipeline usable content
with zero downloads. ``seed_if_empty`` is called at runtime for the in-memory
store so RAG works out of the box even when a persistent ChromaDB is not
available; ``scripts/ingest.py`` writes the same notes to disk and ingests them.
"""

from __future__ import annotations

from typing import Dict

from app.rag.embeddings import Embedder
from app.rag.ingestion import chunk_text
from app.rag.vector_store import StoredDocument, VectorStore

SEED: Dict[str, dict] = {
    "physics_kinematics": {
        "subject": "physics", "chapter": "Kinematics", "grade": "college",
        "text": (
            "Kinematics describes motion without asking what causes it. The key "
            "quantities are displacement, velocity and acceleration. Velocity is the "
            "rate of change of position with time; acceleration is the rate of change "
            "of velocity with time. For constant acceleration, v = v0 + a t, and the "
            "displacement is x = x0 + v0 t + one half a t squared."
        ),
    },
    "physics_newton_laws": {
        "subject": "physics", "chapter": "Newton's Laws", "grade": "college",
        "text": (
            "Newton's first law: an object stays at rest or in uniform motion unless "
            "acted on by a net external force. Newton's second law: the net force on "
            "an object equals its mass times its acceleration, F = m a, with the "
            "acceleration in the direction of the net force. Newton's third law: for "
            "every action there is an equal and opposite reaction."
        ),
    },
    "physics_energy": {
        "subject": "physics", "chapter": "Energy", "grade": "college",
        "text": (
            "Kinetic energy is the energy of motion and equals one half m v squared. "
            "Gravitational potential energy is m g h. The law of conservation of "
            "energy states that energy is neither created nor destroyed, only "
            "transformed. In an isolated system the total mechanical energy is "
            "constant when only conservative forces act."
        ),
    },
    "chemistry_atoms": {
        "subject": "chemistry", "chapter": "Atomic Structure", "grade": "high_school",
        "text": (
            "An atom has a nucleus of protons and neutrons surrounded by electrons. "
            "The atomic number is the number of protons and identifies the element; "
            "carbon has atomic number 6 and oxygen 8. A molecule is two or more atoms "
            "bonded together. An ionic bond forms when electrons transfer between "
            "atoms, producing oppositely charged ions that attract; a covalent bond "
            "forms when atoms share electrons."
        ),
    },
    "chemistry_stoichiometry": {
        "subject": "chemistry", "chapter": "Stoichiometry", "grade": "college",
        "text": (
            "A mole is a fixed number of particles used to count atoms and molecules. "
            "Molar mass is the mass in grams per mole; water is about 18 grams per "
            "mole, so 36 grams of water is 2 moles. Balancing a chemical equation "
            "makes the number of atoms of each element equal on both sides: two "
            "hydrogen molecules react with one oxygen molecule to form two water "
            "molecules."
        ),
    },
    "biology_cell": {
        "subject": "biology", "chapter": "The Cell", "grade": "high_school",
        "text": (
            "The cell is the basic unit of life. Mitochondria produce usable chemical "
            "energy through respiration. Photosynthesis is the process by which plants "
            "convert light energy into chemical energy stored in sugars; it occurs "
            "mainly in the chloroplasts and uses carbon dioxide and water, releasing "
            "oxygen."
        ),
    },
    "math_algebra": {
        "subject": "math", "chapter": "Linear Equations", "grade": "high_school",
        "text": (
            "To solve a linear equation, isolate the variable by applying inverse "
            "operations to both sides. To solve 2 x + 5 = 15, subtract 5 from both "
            "sides to get 2 x = 10, then divide both sides by 2 to get x = 5. The "
            "slope of a line is the change in y divided by the change in x between "
            "two points."
        ),
    },
}


def seed_documents(embedder: Embedder, store: VectorStore, chunk_size: int = 500, overlap: int = 50) -> int:
    """Embed and add the seed corpus to a store. Returns the chunk count."""
    batch = []
    for key, meta in SEED.items():
        base = {"source": key, "subject": meta["subject"], "chapter": meta["chapter"], "grade": meta["grade"]}
        chunks = chunk_text(meta["text"], chunk_size, overlap, base)
        vectors = embedder.embed_documents([c.text for c in chunks])
        for i, (chunk, vec) in enumerate(zip(chunks, vectors)):
            batch.append(StoredDocument(id=f"{key}_{i}", text=chunk.text, embedding=vec, metadata=chunk.metadata))
    store.add(batch)
    return len(batch)


def seed_if_empty(embedder: Embedder, store: VectorStore) -> int:
    """Seed the store only if it is currently empty. Returns chunks added."""
    try:
        if store.count() > 0:
            return 0
    except Exception:  # noqa: BLE001
        return 0
    return seed_documents(embedder, store)
