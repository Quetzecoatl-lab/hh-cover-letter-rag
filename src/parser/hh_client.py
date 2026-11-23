import asyncio
import json
import re
from pathlib import Path
from typing import List, Dict, Any
from urllib.parse import urljoin
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright, Page
from pydantic import BaseModel, Field


class VacancySchema(BaseModel):
    id: str
    search_query: str
    title: str
    company: str
    url: str
    description: str
    key_skills: List[str] = Field(default_factory=list)
    experience: str


class HHPlaywrightScraper:
    BASE_DOMAIN = "https://hh.ru"

    async def scroll_page(self, page: Page, steps: int = 5):
        """Плавная прокрутка страницы вниз для подгрузки карточек."""
        for _ in range(steps):
            await page.evaluate("window.scrollBy(0, 1000)")
            await asyncio.sleep(0.5)

    async def extract_links_from_serp(self, page: Page) -> List[str]:
        """Собирает ссылки на вакансии из текущей страницы поисковой выдачи."""
        await self.scroll_page(page, steps=5)
        raw_hrefs = await page.eval_on_selector_all(
            'a[data-qa="serp-item__title"], a[href*="/vacancy/"]',
            "els => els.map(e => e.getAttribute('href'))"
        )
        urls = []
        for href in raw_hrefs:
            if not href:
                continue
            clean_href = href.split("?")[0]
            if re.search(r"/vacancy/\d+$", clean_href):
                full_url = urljoin(self.BASE_DOMAIN, clean_href)
                if full_url not in urls:
                    urls.append(full_url)
        return urls

    async def search_and_collect_urls(self, page: Page, query: str, pages: int = 2) -> List[str]:
        """Вводит запрос в поисковую строку HH как пользователь и листает страницы."""
        await page.goto(self.BASE_DOMAIN, wait_until="domcontentloaded")
        await asyncio.sleep(2)

        search_input = page.locator('input[data-qa="search-input"], input#a11y-search-input').first
        try:
            await search_input.wait_for(state="visible", timeout=10000)
        except Exception:
            print("\n[!] Не видно строку поиска (пройдите проверку/капчу или залогиньтесь в открытом окне браузера)...")
            await search_input.wait_for(state="visible", timeout=120000)

        await search_input.fill("")
        await search_input.type(query, delay=80)
        await page.keyboard.press("Enter")

        try:
            await page.wait_for_url("**/search/vacancy**", timeout=15000)
        except Exception:
            print(f"[Warning] Не удалось перейти в поиск по запросу '{query}'. Текущий URL: {page.url}")
            return []

        await asyncio.sleep(2)
        collected_urls = []

        for p in range(pages):
            page_urls = await self.extract_links_from_serp(page)
            for u in page_urls:
                if u not in collected_urls:
                    collected_urls.append(u)

            if p < pages - 1:
                next_btn = page.locator('a[data-qa="pager-next"]').first
                if await next_btn.count() > 0:
                    await next_btn.click()
                    await asyncio.sleep(2.5)
                else:
                    break

        return collected_urls

    async def parse_vacancy(self, page: Page, url: str, query: str) -> Dict[str, Any] | None:
        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=20000)
            await page.wait_for_selector('[data-qa="vacancy-title"], h1', timeout=10000)
            await asyncio.sleep(0.7)

            html = await page.content()
            soup = BeautifulSoup(html, "html.parser")

            id_match = re.search(r"/vacancy/(\d+)", url)
            vac_id = id_match.group(1) if id_match else url

            title_el = soup.find(attrs={"data-qa": "vacancy-title"}) or soup.find("h1")
            if not title_el:
                return None
            title = title_el.get_text(strip=True)

            company_el = soup.find(attrs={"data-qa": "vacancy-company-name"})
            company = company_el.get_text(" ", strip=True) if company_el else "Не указана"

            exp_el = soup.find(attrs={"data-qa": "vacancy-experience"})
            experience = exp_el.get_text(strip=True) if exp_el else "Не указан"

            desc_el = soup.find(attrs={"data-qa": "vacancy-description"})
            description = desc_el.get_text(" ", strip=True) if desc_el else ""
            description = re.sub(r"\s+", " ", description).strip()

            skill_els = soup.find_all(attrs={"data-qa": "skills-element"})
            if not skill_els:
                skill_els = soup.find_all(attrs={"data-qa": "bloko-tag__text"})
            key_skills = [s.get_text(strip=True) for s in skill_els if s.get_text(strip=True)]

            if len(description) < 150:
                return None

            vacancy = VacancySchema(
                id=vac_id,
                search_query=query,
                title=title,
                company=company,
                url=url,
                description=description[:3500],
                key_skills=list(dict.fromkeys(key_skills))[:15],
                experience=experience,
            )
            return vacancy.model_dump()
        except Exception:
            return None

    async def run(self, queries: List[str], pages_per_query: int = 2, max_vacancies: int = 100) -> List[Dict[str, Any]]:
        results = []
        user_data_dir = Path("data/browser_profile")
        user_data_dir.mkdir(parents=True, exist_ok=True)

        async with async_playwright() as p:
            context = await p.chromium.launch_persistent_context(
                user_data_dir=str(user_data_dir),
                headless=False,
                args=["--disable-blink-features=AutomationControlled"],
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
                viewport={"width": 1280, "height": 900},
                locale="ru-RU",
            )
            page = context.pages[0] if context.pages else await context.new_page()

            url_to_query: Dict[str, str] = {}
            for query in queries:
                urls = await self.search_and_collect_urls(page, query, pages=pages_per_query)
                for u in urls:
                    if u not in url_to_query:
                        url_to_query[u] = query
                print(f"По ключу '{query}' найдено свежих вакансий: {len(urls)}")

            items_to_parse = list(url_to_query.items())[:max_vacancies]
            print(f"\nНачинаем парсинг карточек (всего {len(items_to_parse)} шт.)...")

            for idx, (url, query) in enumerate(items_to_parse, start=1):
                data = await self.parse_vacancy(page, url, query)
                if data:
                    results.append(data)
                if idx % 10 == 0 or idx == len(items_to_parse):
                    print(f"Обработано {idx}/{len(items_to_parse)} | Успешно сохранено: {len(results)}")

            await context.close()
        return results


async def main():
    queries = [
        "Python разработчик",
        "Machine Learning Engineer",
        "Data Scientist",
    ]
    scraper = HHPlaywrightScraper()
    vacancies = await scraper.run(queries=queries, pages_per_query=2, max_vacancies=100)

    output_path = Path("data/raw/hh_vacancies.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for item in vacancies:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\nГотово! Спарсено {len(vacancies)} живых вакансий с HH.ru в {output_path.resolve()}")


if __name__ == "__main__":
    asyncio.run(main())