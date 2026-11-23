import os
import json
import time
import random
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm
from groq import Groq

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# available_models = [
#     m.id for m in client.models.list().data
#     if "whisper" not in m.id.lower() and "guard" not in m.id.lower() and "tts" not in m.id.lower()
# ]
# # Предпочитаем крупные модели (70b / llama / qwen / gpt-oss), если есть
# preferred = [m for m in available_models if any(k in m.lower() for k in ["70b", "llama", "qwen", "gpt"])]
# MODEL_NAME = preferred if preferred else available_models
# print(f"[OK] Выбрана модель Groq: {MODEL_NAME}")

MODEL_NAME = "qwen/qwen3.8-27b"
def get_working_free_model() -> str:
    """Автоматически находит рабочую бесплатную модель на OpenRouter."""
    print("Поиск доступной бесплатной модели на OpenRouter...")
    resp = requests.get(f"{OPENROUTER_BASE}/models", timeout=15).json()
    free_models = [
        m["id"] for m in resp.get("data", [])
        if ":free" in m["id"] and "vision" not in m["id"].lower() and "image" not in m["id"].lower()
    ]
    # Добавляем универсальный роутер бесплатных моделей в начало списка
    candidates = ["openrouter/free"] + free_models[:15]

    for model_id in candidates:
        try:
            test = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": "Ответь одним словом: Привет"}],
                max_tokens=10,
            )
            if test.choices and next(iter(test.choices)).message.content:
                print(f"[OK] Выбрана рабочая модель: {model_id}")
                return model_id
        except Exception as e:
            print(f"[Пропуск] {model_id}: {e}")
            continue

    raise RuntimeError("Не удалось найти доступную бесплатную модель на OpenRouter.")


PROFILES = {
    "Python / Backend": (
        "Python Backend разработчик (опыт 4 года). Стек: Python, FastAPI, Django, PostgreSQL, Redis, RabbitMQ, Docker. "
        "Кейс: перевел монолит на асинхронные микросервисы (нагрузка до 3000 RPS), оптимизировал тяжелые SQL-запросы, "
        "ускорив ответы API в 4 раза."
    ),
    "ML / AI / NLP": (
        "Machine Learning / NLP инженер (опыт 3 года). Стек: Python, PyTorch, Transformers, LangChain, Qdrant, QLoRA, vLLM. "
        "Кейс: внедрил RAG-систему для службы поддержки, автоматизировав 35% обращений, и дообучил открытую LLM (Qwen-7B) "
        "для генерации деловых ответов без галлюцинаций."
    ),
    "Data Science / Analytics": (
        "Data Scientist / Продуктовый аналитик (опыт 3.5 года). Стек: Python, SQL, ClickHouse, Airflow, CatBoost, A/B-тесты. "
        "Кейс: построил модель прогноза оттока (ROC-AUC 0.86) и провел серию A/B-тестов, что увеличило retention пользователей на 6%."
    ),
}

SYSTEM_PROMPT = """Ты — карьерный консультант, который пишет сопроводительные письма напрямую для CEO и фаундеров IT-компаний.

Требования к письму:
1. Объем: от 400 до 900 символов. Коротко и по делу.
2. Структура:
   - Первое предложение: суть бизнес-задачи из описания вакансии.
   - 2-3 предложения: конкретный опыт и цифры из профиля кандидата, которые решают эту задачу.
   - Последнее предложение: короткий призыв созвониться на 15 минут.
3. Запрещенные штампы: "Доброго времени суток", "Меня заинтересовала ваша вакансия", "Прошу рассмотреть мое резюме", "С уважением".
4. Строго запрещено придумывать навыки или кейсы, которых нет в профиле кандидата.
5. Выдай ТОЛЬКО готовый текст письма на русском языке без заголовков."""

BAD_PHRASES = [
    "доброго времени суток",
    "меня заинтересовала ваша вакансия",
    "прошу рассмотреть мое резюме",
    "с уважением",
    "надеюсь на скорый ответ",
]


def is_valid_text(letter: str) -> bool:
    if not (300 <= len(letter) <= 1200):
        return False
    low = letter.lower()
    if any(phrase in low for phrase in BAD_PHRASES):
        return False
    if "**" in letter or "```" in letter:
        return False
    return True


def check_hallucinations(model_name: str, profile_text: str, letter: str) -> bool:
    prompt = (
        f"Проверь, упоминаются ли в письме технологии или достижения, которых НЕТ в профиле кандидата.\n"
        f"Профиль кандидата: {profile_text}\n"
        f"Письмо: {letter}\n"
        f"Ответь одним словом: YES (если всё строго по профилю) или NO (если есть выдумки)."
    )
    try:
        res = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=10,
        )
        answer = next(iter(res.choices)).message.content.strip().upper()
        return "YES" in answer
    except Exception:
        return False


def main():
    model_name = MODEL_NAME

    excel_path = Path("data/processed/hh_vacancies_report.xlsx")
    df = pd.read_excel(excel_path)

    relevant_df = df[df["Статус релевантности"] == "Релевантна"].copy()
    print(f"Найдено релевантных вакансий для генерации: {len(relevant_df)}")

    dataset = []

    for _, row in tqdm(relevant_df.iterrows(), total=len(relevant_df), desc="Генерация писем"):
        category = str(row.get("Поисковый ключ (Направление)", "Python / Backend"))
        profile = PROFILES.get(category, PROFILES["Python / Backend"])

        user_prompt = (
            f"Профиль кандидата:\n{profile}\n\n"
            f"Вакансия: {row['Название вакансии']}\n"
            f"Ключевые навыки: {row['Ключевые навыки']}\n"
            f"Описание: {str(row['Описание (фрагмент)'])[:1500]}"
        )

        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.7,
                max_tokens=400,
            )
            letter = next(iter(response.choices)).message.content.strip()
        except Exception as e:
            print(f"\n[Пауза из-за лимита API]: {e}")
            time.sleep(10)
            continue

        time.sleep(2)

        if not is_valid_text(letter):
            continue

        if not check_hallucinations(model_name, profile, letter):
            continue

        dataset.append({
            "instruction": "Напиши короткое сопроводительное письмо для CEO на основе профиля кандидата и вакансии.",
            "input": user_prompt,
            "output": letter,
        })
        time.sleep(4)

    random.shuffle(dataset)
    split_idx = max(1, int(len(dataset) * 0.85))
    train_samples = dataset[:split_idx]
    val_samples = dataset[split_idx:]

    out_dir = Path("data/processed")
    for filename, samples in [("train.jsonl", train_samples), ("val.jsonl", val_samples)]:
        with open(out_dir / filename, "w", encoding="utf-8") as f:
            for item in samples:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\nГотово! Из {len(relevant_df)} вакансий отбор прошли {len(dataset)} писем.")
    print(f"Сохранено: train={len(train_samples)}, val={len(val_samples)}")


if __name__ == "__main__":
    main()
