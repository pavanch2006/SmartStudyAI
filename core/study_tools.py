from core.tutor import ask_tutor

MAX_DIRECT_CHARS = 300_000


def _extract_text(chunk):
    if isinstance(chunk, dict):
        return chunk.get("text", "")
    return str(chunk)


def prepare_material(chunks):
    if not chunks:
        raise ValueError("No document content available.")

    full_text = "\n\n".join(_extract_text(c) for c in chunks)

    if len(full_text) <= MAX_DIRECT_CHARS:
        return full_text

    batch_size = 40
    section_summaries = []

    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start + batch_size]
        context = "\n\n".join(_extract_text(c) for c in batch)

        prompt = f"""
Read the following study material and extract the important information accurately.

STUDY MATERIAL:
{context}

Create a concise section summary.
Preserve important definitions, concepts, formulas and examples.
Do not add information that is not present.
"""
        section_summaries.append(ask_tutor(prompt))

    return "\n\n".join(section_summaries)


def generate_summary(chunks):
    material = prepare_material(chunks)

    prompt = f"""
You are an academic study assistant.

Create a well-organized summary from this material.

Include:
1. Main topics
2. Important concepts
3. Key definitions
4. Important points to remember

Use clear headings and simple language.
Use only information found in the material.

MATERIAL:
{material}
"""
    return ask_tutor(prompt)


def generate_notes(chunks):
    material = prepare_material(chunks)

    prompt = f"""
You are an academic note-making assistant.

Convert the following material into structured revision notes.

Use:
1. Topic headings
2. Subheadings
3. Bullet points
4. Important definitions
5. Examples where available
6. Important formulas, if present

Preserve the original meaning.
Do not invent facts.

STUDY MATERIAL:
{material}
"""
    return ask_tutor(prompt)


def generate_quiz(chunks):
    material = prepare_material(chunks)

    prompt = f"""
You are an academic quiz generator.

Generate 5 multiple-choice questions based ONLY on the supplied study material.

For every question provide:
- Question
- Four options (A, B, C, D)
- Correct answer
- Short explanation

Include a mixture of easy and moderate questions.

Do not generate questions whose answers are not supported by the material.

STUDY MATERIAL:
{material}
"""
    return ask_tutor(prompt)