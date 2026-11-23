import json
import re
from pathlib import Path
import pandas as pd


# Целевые ключевые слова для оценки релевантности (например, под IT / Python / ML / Data)
TARGET_KEYWORDS = {
    "Python / Backend": [r"\bpython\b", r"\bbackend\b", r"\bfastapi\b", r"\bdjango\b", r"бэкенд"],
    "ML / AI / NLP": [r"\bml\b", r"machine learning", r"\bnlp\b", r"\bllm\b", r"\bai\b", r"машинн\w+ обучен"],
    "Data Science / Analytics": [r"data scien", r"data analy", r"аналитик данных", r"\bds\b", r"\bsql\b"],
    "Frontend": [r"\bfrontend\b", r"\breact\b", r"\bvue\b", r"\btypescript\b", r"фронтенд"],
    "Product / Project IT": [r"product manager", r"продакт", r"менеджер продукт"],
}

# Стоп-слова для отсева нерелевантных вакансий (не IT или неподходящий профиль)
EXCLUDE_KEYWORDS = [
    r"менеджер по продажам", r"водитель", r"курьер", r"кладовщик", r"бухгалтер",
    r"1с", r"1c", r"преподаватель", r"репетитор", r"администратор"
]


def evaluate_relevance(row: dict) -> tuple[str, str, str]:
    """Определяет поисковый ключ, статус релевантности и обоснование."""
    title = str(row.get("title", "")).lower()
    desc = str(row.get("description", "")).lower()
    skills = " ".join(row.get("key_skills", [])).lower()
    full_text = f"{title} {skills} {desc[:600]}"

    # 1. Проверка на стоп-слова в названии
    for bad_kw in EXCLUDE_KEYWORDS:
        if re.search(bad_kw, title):
            return "Не определено", "Не релевантна", f"Стоп-слово в названии ({bad_kw})"

    # 2. Поиск совпадений с целевыми ключами
    matched_category = None
    for category, patterns in TARGET_KEYWORDS.items():
        if any(re.search(p, title) for p in patterns):
            matched_category = category
            break

    if matched_category:
        return matched_category, "Релевантна", "Точное совпадение профиля в названии вакансии"

    # Если в названии нет, но есть в навыках/описании
    for category, patterns in TARGET_KEYWORDS.items():
        if any(re.search(p, full_text) for p in patterns):
            return category, "Условно релевантна", "Совпадение по стеку в описании/навыках"

    return "Прочее", "Не релевантна", "Не относится к целевому IT-стеку"


def main():
    input_path = Path("data/raw/hh_vacancies.jsonl")
    output_excel = Path("data/processed/hh_vacancies_report.xlsx")
    output_excel.parent.mkdir(parents=True, exist_ok=True)

    with open(input_path, "r", encoding="utf-8") as f:
        raw_data = [json.loads(line) for line in f]

    rows = []
    for item in raw_data:
        key_query, status, reason = evaluate_relevance(item)
        rows.append({
            "ID вакансии": item.get("id"),
            "Поисковый ключ (Направление)": key_query,
            "Статус релевантности": status,
            "Причина оценки": reason,
            "Название вакансии": item.get("title"),
            "Компания": item.get("company"),
            "Опыт": item.get("experience"),
            "Ключевые навыки": ", ".join(item.get("key_skills", [])),
            "Ссылка на HH": item.get("url"),
            "Описание (фрагмент)": item.get("description", "")[:500] + "...",
        })

    df = pd.DataFrame(rows)
    # Сортируем так, чтобы релевантные были сверху
    df.sort_values(by=["Статус релевантности", "Поисковый ключ (Направление)"], ascending=[False, True], inplace=True)

    df.to_excel(output_excel, index=False, engine="openpyxl")

    print(f"Таблица сохранена: {output_excel.resolve()}")
    print("\nСтатистика релевантности:")
    print(df["Статус релевантности"].value_counts())


if __name__ == "__main__":
    main()