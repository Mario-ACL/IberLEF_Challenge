# En YUCA: usa el conda env 'iberlef' (ya tiene todo).
# En Colab descomenta la línea siguiente:
# !pip install -q sentence-transformers accelerate

import os
os.environ.setdefault("HF_HOME", os.environ.get("SCRATCH", "/tmp") + "/hf_cache")

import gc
import json
import random
import re

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE, torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")

path = r"./Data/raw"

df_info = pd.read_csv(f"{path}/INFO_AUG.csv")
df_quest = pd.read_csv(f"{path}/QUEST_AUG.csv")
df_reflex = pd.read_csv(f"{path}/REFLEX_AUG.csv")

df_info.head()

df_info.shape

df_info['behavior_code_1'].value_counts()

df_info['client_utterance'].nunique()

df_info.iloc[0]['clinician_utterance']

df_info.iloc[1]['clinician_utterance']

df_quest.head()

df_quest.shape

df_quest['behavior_code_1'].value_counts()

df_quest['client_utterance'].nunique()

df_reflex.head()

df_reflex.shape

df_reflex['behavior_code_1'].value_counts()

df_reflex.isna().sum()

MODEL_NAME = "./models/qwen2.5-14b"
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.9
TOP_P = 0.95
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# bf16 funciona en MI210 (CDNA2) y en GPUs NVIDIA Ampere/Hopper.
# En T4 (Turing) bf16 NO está soportado: ahí usarías float16 + 4-bit.
DTYPE = torch.bfloat16

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype=DTYPE,
)
model.eval()
print("Model loaded")

print(f"Footprint: {model.get_memory_footprint() / 1e9:.2f} GB")
print(f"VRAM allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

BC_DEFINITIONS = {
    "SR": (
        "Simple Reflection (SR): el clínico repite, parafrasea o refleja literalmente "
        "lo que el paciente acaba de decir SIN añadir significado, inferencia ni énfasis. "
        "Reproduce el contenido pero no profundiza en él. Suele empezar con \"Entonces...\", "
        "\"Te escucho decir que...\", \"Estás diciendo que...\"."
    ),
    "CR": (
        "Complex Reflection (CR): el clínico añade significado, énfasis o interpretación a "
        "lo que el paciente dijo. Infiere emociones no verbalizadas, identifica ambivalencia, "
        "reformula con metáforas o conecta puntos. Va MÁS ALLÁ del contenido literal."
    ),
    "OQ": (
        "Open Question (OQ): pregunta abierta que no se puede responder con sí/no ni con "
        "un dato corto. Invita a elaborar, explorar, narrar. Suele empezar con \"qué\", "
        "\"cómo\", \"de qué manera\", \"cuéntame...\"."
    ),
    "CQ": (
        "Closed Question (CQ): pregunta cerrada que se responde con sí/no, una opción de un "
        "conjunto pequeño, o un dato concreto (fecha, número, nombre). Limita la respuesta."
    ),
    "GI": (
        "Giving Information (GI): el clínico da información, educación o feedback de manera "
        "NEUTRAL, sin intentar persuadir ni aconsejar. Comparte datos, explica un mecanismo, "
        "informa de un recurso, sin presionar al paciente a cambiar."
    ),
    "PE": (
        "Persuasion (PE): el clínico intenta cambiar la opinión, actitud o conducta del paciente "
        "mediante argumentos, advertencias, consejos no solicitados o apelaciones. NO respeta "
        "plenamente la autonomía del paciente."
    ),
}

PAIRS = {
    "reflex": ("SR", "CR"),
    "quest": ("OQ", "CQ"),
    "info": ("GI", "PE"),
}


DIVERSITY_AXES = {
    "edad": ["18-25", "26-40", "41-60", "60+"],
    "género": ["mujer", "hombre", "no binario"],
    "tema": [
        "consumo de tabaco",
        "consumo de alcohol",
        "consumo de marihuana",
        "consumo de cocaína",
        "trastorno de ansiedad",
        "depresión",
        "dolor crónico",
        "insomnio",
        "trastorno alimentario",
        "sedentarismo y obesidad",
        "diabetes tipo 2 mal controlada",
        "hipertensión",
        "adherencia a medicación antirretroviral",
        "juego patológico",
        "violencia intrafamiliar (víctima)",
        "duelo no resuelto",
    ],
    "etapa_cambio": [
        "precontemplación",
        "contemplación",
        "preparación",
        "acción",
        "mantenimiento",
        "recaída",
    ],
    "profesional": [
        "médico de familia",
        "psicólogo clínico",
        "enfermero/a de atención primaria",
        "trabajador/a social",
        "psiquiatra",
        "nutricionista",
    ],
    "registro": [
        "español neutro",
        "coloquial mexicano",
        "rioplatense",
        "peninsular formal",
        "andino coloquial",
    ],
}

def sample_profile(rng=random):
    return {axis: rng.choice(values) for axis, values in DIVERSITY_AXES.items()}

[sample_profile() for _ in range(3)]

SYSTEM_PROMPT = (
    "Eres un experto clínico en Entrevista Motivacional (MI) y en la rúbrica MITI 4.2.1. "
    "Tu trabajo es generar turnos de conversación sintéticos en español, realistas y "
    "clínicamente plausibles, para entrenar clasificadores de códigos de comportamiento. "
    "Respetas siempre la definición exacta del código objetivo. Devuelves SOLO un JSON "
    "válido, sin texto adicional, sin markdown, sin comentarios."
)

def build_user_prompt(target_code, contrast_code, seed_examples, profile):
    target_def = BC_DEFINITIONS[target_code]
    contrast_def = BC_DEFINITIONS[contrast_code]

    shots = "\n\n".join(
        f"Ejemplo {i+1}:\n"
        f'{{"client_utterance": {json.dumps(row.client_utterance, ensure_ascii=False)}, '
        f'"clinician_utterance": {json.dumps(row.clinician_utterance, ensure_ascii=False)}}}'
        for i, row in enumerate(seed_examples.itertuples())
    )

    profile_text = "\n".join(f"- {k}: {v}" for k, v in profile.items())

    return f"""Genera UN turno de conversación cliente-clínico que ejemplifique de forma INEQUÍVOCA el código **{target_code}**.

Definición del código objetivo ({target_code}):
{target_def}

Para evitar confusión, NO debe encajar en este otro código del par ({contrast_code}):
{contrast_def}

Perfil del paciente y contexto clínico (úsalo como inspiración, no lo cites literalmente):
{profile_text}

Ejemplos de referencia (mismo código objetivo):
{shots}

Restricciones:
- `client_utterance`: 1-3 oraciones, voz del paciente, coherente con el perfil.
- `clinician_utterance`: 1-2 oraciones, debe ejemplificar {target_code} estrictamente.
- Evita copiar frases de los ejemplos. Varía el vocabulario, los arranques y la estructura.
- No incluyas el nombre del código en el texto.

Devuelve SOLO el JSON con las claves `client_utterance` y `clinician_utterance`."""


JSON_RE = re.compile(r"\{[\s\S]*?\}")

def parse_response(text):
    match = JSON_RE.search(text)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    if "client_utterance" not in obj or "clinician_utterance" not in obj:
        return None
    client = str(obj["client_utterance"]).strip()
    clinician = str(obj["clinician_utterance"]).strip()
    if not client or not clinician:
        return None
    return {"client_utterance": client, "clinician_utterance": clinician}

def generate_one(target_code, contrast_code, seed_df, n_shots=3):
    profile = sample_profile()
    pool = seed_df[seed_df["behavior_code_1"] == target_code]
    shots = pool.sample(n=min(n_shots, len(pool)))

    user_prompt = build_user_prompt(target_code, contrast_code, shots, profile)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    chat_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(chat_text, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = out[0][inputs.input_ids.shape[1]:]
    decoded = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return parse_response(decoded), decoded


parsed, raw = generate_one("SR", "CR", df_reflex)
print(raw)
parsed

parsed, _ = generate_one("CR", "SR", df_reflex)
parsed

def generate_for_class(target_code, contrast_code, seed_df, n_target, max_attempts=None):
    if max_attempts is None:
        max_attempts = n_target * 3
    rows, attempts = [], 0
    pbar = tqdm(total=n_target, desc=f"{target_code}")
    while len(rows) < n_target and attempts < max_attempts:
        attempts += 1
        parsed, _ = generate_one(target_code, contrast_code, seed_df)
        if parsed is None:
            continue
        rows.append({**parsed, "behavior_code_1": target_code})
        pbar.update(1)
    pbar.close()
    print(f"  {target_code}: {len(rows)}/{n_target} en {attempts} intentos")
    return pd.DataFrame(rows)


from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer("./models/minilm", device=str(DEVICE))
print("Embedder loaded")

def deduplicate_semantic(df, threshold=0.92):
    if len(df) == 0:
        return df
    texts = (df["client_utterance"] + " " + df["clinician_utterance"]).tolist()
    embs = embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    keep_idx, keep_embs = [], []
    for i, e in enumerate(embs):
        if not keep_embs:
            keep_idx.append(i); keep_embs.append(e); continue
        sims = np.array(keep_embs) @ e
        if sims.max() < threshold:
            keep_idx.append(i); keep_embs.append(e)
    return df.iloc[keep_idx].reset_index(drop=True)

def balance_classes(df, classes, n_per_class):
    parts = []
    for c in classes:
        sub = df[df["behavior_code_1"] == c]
        if len(sub) < n_per_class:
            print(f"  ⚠ clase {c} solo tiene {len(sub)} tras dedupe")
        parts.append(sub.head(n_per_class))
    return pd.concat(parts, ignore_index=True).sample(frac=1, random_state=SEED).reset_index(drop=True)

OVER_GENERATE = 130
TARGET_PER_CLASS = 100

raw_sr = generate_for_class("SR", "CR", df_reflex, n_target=OVER_GENERATE)
raw_cr = generate_for_class("CR", "SR", df_reflex, n_target=OVER_GENERATE)
raw_reflex = pd.concat([raw_sr, raw_cr], ignore_index=True)

raw_reflex.shape

raw_reflex['behavior_code_1'].value_counts()

raw_reflex.head()

deduped_reflex = pd.concat(
    [deduplicate_semantic(raw_reflex[raw_reflex['behavior_code_1'] == c]) for c in ("SR", "CR")],
    ignore_index=True,
)
deduped_reflex['behavior_code_1'].value_counts()

final_reflex = balance_classes(deduped_reflex, classes=("SR", "CR"), n_per_class=TARGET_PER_CLASS)
final_reflex.shape

final_reflex.head()

final_reflex.iloc[0]['clinician_utterance']

final_reflex.to_csv("generated_dataset_reflex.csv", index=False, encoding="utf-8-sig")
print(f"Saved {len(final_reflex)} rows")

raw_oq = generate_for_class("OQ", "CQ", df_quest, n_target=OVER_GENERATE)
raw_cq = generate_for_class("CQ", "OQ", df_quest, n_target=OVER_GENERATE)
raw_quest = pd.concat([raw_oq, raw_cq], ignore_index=True)

raw_quest.shape

raw_quest['behavior_code_1'].value_counts()

raw_quest.head()

deduped_quest = pd.concat(
    [deduplicate_semantic(raw_quest[raw_quest['behavior_code_1'] == c]) for c in ("OQ", "CQ")],
    ignore_index=True,
)
deduped_quest['behavior_code_1'].value_counts()

final_quest = balance_classes(deduped_quest, classes=("OQ", "CQ"), n_per_class=TARGET_PER_CLASS)
final_quest.shape

final_quest.head()

final_quest.to_csv("generated_dataset_quest.csv", index=False, encoding="utf-8-sig")
print(f"Saved {len(final_quest)} rows")

raw_gi = generate_for_class("GI", "PE", df_info, n_target=OVER_GENERATE)
raw_pe = generate_for_class("PE", "GI", df_info, n_target=OVER_GENERATE)
raw_info = pd.concat([raw_gi, raw_pe], ignore_index=True)

raw_info.shape

raw_info['behavior_code_1'].value_counts()

raw_info.head()

deduped_info = pd.concat(
    [deduplicate_semantic(raw_info[raw_info['behavior_code_1'] == c]) for c in ("GI", "PE")],
    ignore_index=True,
)
deduped_info['behavior_code_1'].value_counts()

final_info = balance_classes(deduped_info, classes=("GI", "PE"), n_per_class=TARGET_PER_CLASS)
final_info.shape

final_info.head()

final_info.to_csv("generated_dataset_info.csv", index=False, encoding="utf-8-sig")
print(f"Saved {len(final_info)} rows")

del model, tokenizer, embedder
gc.collect()
torch.cuda.empty_cache()
print("GPU memory cleared")
