import wikipediaapi
from pathlib import Path

USER_AGENT = "rag-milvus-tester/0.1 (contact: your@email.com)"

wiki = wikipediaapi.Wikipedia(language='en', user_agent=USER_AGENT)
topics = [
    "Auto mechanic",
    "Automobile repair shop",
    "Car maintenance",
    "Vehicle inspection",
    "Automotive repair technician"
]

data_dir = Path("data")
data_dir.mkdir(exist_ok=True)

for topic in topics:
    page = wiki.page(topic)
    if page.exists():
        filename = data_dir / f"{topic.replace(' ', '_')}.txt"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(page.text)
        print(f"Saved: {filename}")
    else:
        print(f"Page not found: {topic}")
