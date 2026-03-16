import requests
from bs4 import BeautifulSoup
import json
import re
import os
import time
from urllib.parse import quote

class LotrFandomProcessor:
    def __init__(self):
        self.base_url = "https://lotr.fandom.com/wiki/"
        self.knowledge_base_dir = "knowledge_base"
        self.terms_map = {}
        self.pages_to_scrape = [
            # Персонажи
            "Aragorn_II", "Gandalf", "Frodo_Baggins", "Samwise_Gamgee",
            "Legolas", "Gimli", "Boromir", "Meriadoc_Brandybuck",
            "Peregrin_Took", "Elrond", "Galadriel", "Arwen",
            "Sauron", "Saruman", "Gollum", "Nazgûl",

            # Расы и народы
            "Hobbits", "Elves", "Dwarves", "Men",
            "Orcs", "Ents", "Balrog",

            # Места
            "The_Shire", "Rivendell", "Mordor", "Gondor",
            "Rohan", "Moria", "Lothlórien", "Minas_Tirith",

            # Объекты и артефакты
            "One_Ring", "Sting", "Andúril", "Palantír",
            "Silmarils", "Mithril"
        ]

    def create_custom_terms_map(self):
        """Создание словаря замен для вселенной Властелина Колец"""
        self.terms_map = {
        # === MAIN CHARACTERS ===
        "Frodo Baggins": "Kaelen Mosswood",
        "Samwise Gamgee": "Roderick Greenhand",
        "Meriadoc Brandybuck": "Julian Riverbend",
        "Peregrin Took": "Benedict Hilltop",
        "Bilbo Baggins": "Arthur Oldbarrel",
        "Gollum": "Slink",
        "Smeagol": "Glint",
        "Frodo" : "Kaelen",
        "Isildur": "Forman",

        "Aragorn": "Thorvald Stormcrown",
        "Boromir": "Cassian Steelheart",
        "Faramir": "Lucian Steelheart",
        "Denethor": "Marcus Firebrand",
        "Theoden": "Roderick Goldmane",
        "Eomer": "Siegfried Horselord",
        "Eowyn": "Isolde Shieldmaiden",

        "Gandalf": "Orin Greybeard",
        "Saruman": "Malachi Whitehand",
        "Sauron": "Morghul",
        "Witch-king": "Dreadlord",

        "Legolas": "Aelar Swiftarrow",
        "Elrond": "Eldrin Half-Elven",
        "Galadriel": "Lyra Silverwood",
        "Arwen": "Sylvana Evenstar",
        "Celeborn": "Theron Wiseoak",

        "Gimli": "Borin Stoneheart",
        "Thorin Oakenshield": "Durin Ironwill",
        "Balin": "Fundin Deepdelver",

        # === RACES AND PEOPLES ===
        "Hobbits": "Halflings",
        "Elves": "Eldar",
        "Dwarves": "Stonefolk",
        "Men": "Sunfolk",
        "Orcs": "Grimkin",
        "Uruk-hai": "Blackspawn",
        "Ents": "Treeherders",
        "Trolls": "Stone-trolls",
        "Eagles": "Sky-lords",
        "Wizards": "Lore-masters",
        "Nazgul": "Shadow-wraiths",

        # === LOCATIONS AND REALMS ===
        "Middle-earth": "Aetheria",
        "The Shire": "Greenhaven",
        "Mordor": "Shadowland",
        "Gondor": "Sunstone Kingdom",
        "Rohan": "Horse-plains",
        "Arnor": "North-realm",
        "Mirkwood": "Darkwood",
        "Fangorn": "Elderwood",

        "Minas Tirith": "White Spire",
        "Edoras": "Golden Hall",
        "Helm's Deep": "Stone Gorge",
        "Isengard": "Wizard's Spire",
        "Rivendell": "Valley Haven",
        "Lothlorien": "Golden Grove",
        "Moria": "Deep-delving",
        "Bree": "Crossroads Town",
        "Minas Morgul": "Ghost City",

        "Misty Mountains": "Cloud-peaks",
        "Lonely Mountain": "Sole Peak",
        "Mount Doom": "Fire Mountain",
        "Dead Marshes": "Blight Fens",
        "Old Forest": "Ancient Wood",

        # === ARTIFACTS AND OBJECTS ===
        "One Ring": "Soul-band",
        "Three Rings": "Elven Bands",
        "Seven Rings": "Stone Bands",
        "Nine Rings": "Mortal Bands",

        "Sting": "Glimmer",
        "Anduril": "Sunblade",
        "Glamdring": "Foe-cleaver",
        "Orcrist": "Goblin-bane",
        "Narsil": "Star-blade",

        "Palantir": "Far-seer",
        "Silmarils": "Star-gems",
        "Arkenstone": "Mountain-heart",
        "Mithril": "Star-metal",
        "Phial of Galadriel": "Star-light",

        # === EVENTS AND CONCEPTS ===
        "The Fellowship": "The Company",
        "The War of the Ring": "The Shadow War",
        "The Third Age": "The Bronze Era",
        "The White Council": "The Sage Council",
        "The Doom of Mandos": "The Elder Curse",

        # === DEITIES AND SPIRITS ===
        "Eru Iluvatar": "The First",
        "Valar": "The Watchers",
        "Maiar": "The Shapers",
        "Morgoth": "The Void",
        "Manwe": "Sky-father",
        "Varda": "Star-queen",
        "Ulmo": "Sea-lord",

        # === LANGUAGES AND WRITING ===
        "Quenya": "High-tongue",
        "Sindarin": "Wood-tongue",
        "Khuzdul": "Stone-speech",
        "Black Speech": "Dark-tongue",
        "Tengwar": "Elder-letters",
        "Cirth": "Stone-runes",

        # === CULTURE AND SOCIETY ===
        "Pipe-weed": "Hearth-leaf",
        "Lembas": "Way-bread",
        "Miruvor": "Sun-draught",
        "The Red Book": "The Chronicles",

        # === ADDITIONAL TERMS ===
        "Shelob": "Venom-weaver",
        "Watcher in the Water": "Lake-guardian",
        "Great Eagles": "Storm-riders",
        "Dragons": "Fire-wyrms",

        "Anduin": "Great River",
        "Bruinen": "Swift-water",
        "Celebrant": "Silver-stream",
        "Entwash": "Wood-river",

        "First Age": "Dawn Era",
        "Second Age": "Stone Era",
        "Years of the Trees": "Light Years",

        "House of Elrond": "Line of Eldrin",
        "House of Durin": "Stone Lineage",
        "Dunedain": "West-men",
        "Rohirrim": "Plains-riders",

        "Elven-craft": "Star-work",
        "Dwarven-forged": "Deep-forged",
        "Seeing-stones": "Gaze-crystals",
        "Mithril-mail": "Star-armor",

        "The Paths of the Dead": "Ghost Roads",
        "Dimrill Dale": "Shadow Vale",
        "Glittering Caves": "Crystal Caverns",
        "Westernesse": "Sunset-land",
        "Undying Lands": "Eternal Shores",
        "The Straight Road": "Star-path",
        "Doors of Durin": "West-gate",
        "Bridge of Khazad-dum": "Stone Bridge",
        "Prancing Pony": "Green Dragon",
        "Bag End": "Hill-home",
        "Green Dragon": "Golden Perch",

        "Elven-king": "Star-lord",
        "Dwarf-lord": "Stone-chief",
        "Orc-chief": "Shadow-captain",
        "Ring-lore": "Band-craft",
        "Shadowfax": "Storm-runner",
        "Bill the Pony": "Faithful Steed",
        "Tom Bombadil": "Wood-master",
        "Goldberry": "River-daughter",
        "Barrow-wights": "Grave-spirits",
        "Old Man Willow": "Ancient Willow",
        "The Withywindle": "Enchanted Stream",

        # === FILM-RELATED ===
        "Peter Jackson": "Alistair Finch",
        "Viggo Mortensen": "Erik Stormborn",
        "Ian McKellen": "Malcolm Greymantle",
        "Elijah Wood": "Julian Mosswood",
        "Sean Astin": "Thomas Greenfield",
        "Orlando Bloom": "Sebastian Swiftarrow",
        "John Rhys-Davies": "Gareth Ironoak",
        "Sean Bean": "Marcus Steelheart",
        "Christopher Lee": "Alistair Whitehand",
        "Andy Serkis": "Damian Shadowkin",

         "Aragorn": "Thorvald Stormcrown",
         "Boromir": "Cassian Steelheart",
         "Faramir": "Lucian Steelheart",
         "Denethor": "Marcus Firebrand",
         "Theoden": "Roderick Goldmane",
         "Eomer": "Siegfried Horselord",
         "Eowyn": "Isolde Shieldmaiden",

         "Gandalf": "Orin Greybeard",
         "Saruman": "Malachi Whitehand",
         "Sauron": "Morghul",
         "Witch-king": "Dreadlord",

         "Legolas": "Aelar Swiftarrow",
         "Elrond": "Eldrin Half-Elven",
         "Galadriel": "Lyra Silverwood",
         "Arwen": "Sylvana Evenstar",
         "Celeborn": "Theron Wiseoak",
         "Thranduil": "Laeron Woodking",

         "Gimli": "Borin Stoneheart",
         "Thorin Oakenshield": "Durin Ironwill",
         "Balin": "Fundin Deepdelver",
         "Gloin": "Thrain Firebeard",

         # === RACES AND PEOPLES ===
         "Hobbits": "Halflings",
         "Elves": "Eldar",
         "Dwarves": "Stonefolk",
         "Men": "Sunfolk",
         "Orcs": "Grimkin",
         "Uruk-hai": "Blackspawn",
         "Ents": "Treeherders",
         "Trolls": "Stone-trolls",
         "Eagles": "Sky-lords",
         "Wizards": "Lore-masters",
         "Nazgul": "Shadow-wraiths",
         "Sindar": "Eldar of Darkwood",
         "Galadhrim": "Golden Grove Elves",

         # === LOCATIONS AND REALMS ===
         "Middle-earth": "Aetheria",
         "The Shire": "Greenhaven",
         "Mordor": "Shadowland",
         "Gondor": "Sunstone Kingdom",
         "Rohan": "Horse-plains",
         "Arnor": "North-realm",
         "Mirkwood": "Darkwood",
         "Fangorn": "Elderwood",
         "Ithilien": "Moonwood",
         "Aman": "The Undying Lands",
         "Woodland Realm": "Forest Kingdom",
         "Northern Mirkwood": "Northern Darkwood",

         "Minas Tirith": "White Spire",
         "Edoras": "Golden Hall",
         "Helm's Deep": "Stone Gorge",
         "Isengard": "Wizard's Spire",
         "Rivendell": "Valley Haven",
         "Lothlorien": "Golden Grove",
         "Moria": "Deep-delving",
         "Bree": "Crossroads Town",
         "Minas Morgul": "Ghost City",
         "Caradhras": "Red Peak",
         "Khazad-dum": "Dwarf-delving",

         "Misty Mountains": "Cloud-peaks",
         "Lonely Mountain": "Sole Peak",
         "Mount Doom": "Fire Mountain",
         "Dead Marshes": "Blight Fens",
         "Old Forest": "Ancient Wood",
         "Anduin": "Great River",

         # === ARTIFACTS AND OBJECTS ===
         "One Ring": "Soul-band",
         "Three Rings": "Elven Bands",
         "Seven Rings": "Stone Bands",
         "Nine Rings": "Mortal Bands",

         "Sting": "Glimmer",
         "Anduril": "Sunblade",
         "Glamdring": "Foe-cleaver",
         "Orcrist": "Goblin-bane",
         "Narsil": "Star-blade",

         "Palantir": "Far-seer",
         "Silmarils": "Star-gems",
         "Arkenstone": "Mountain-heart",
         "Mithril": "Star-metal",
         "Phial of Galadriel": "Star-light",
         "long white knife": "silver dagger",
         "Galadhrim longbow": "Golden Grove bow",

         # === EVENTS AND CONCEPTS ===
         "The Fellowship": "The Company",
         "The War of the Ring": "The Shadow War",
         "The Third Age": "The Bronze Era",
         "The White Council": "The Sage Council",
         "The Doom of Mandos": "The Elder Curse",
         "Council of Elrond": "Council of Eldrin",
         "Nine Walkers": "Nine Companions",
         "Nine Riders": "Shadow Nine",
         "Battle of the Bridge of Khazad-dum": "Battle of the Stone Bridge",
         "Breaking of the Fellowship": "Parting of the Company",

         # === FILM AND MEDIA ===
         "Donato Giancola": "Artemis Brushwood",
         "Olga Serebryakova": "Irina Silverhand",
         "Khraniteli": "Guardians Saga",
         "LotR films": "Chronicles of Aetheria",
         "video games": "interactive chronicles",

         # === MILITARY AND GROUPS ===
         "éored": "horse troop",
         "fellbeast": "shadow-wing",
         "Wargs": "dire wolves",

         # === TIME AND ERA ===
         "Fourth Age": "New Era",
         "FO 120": "NE 120",

         # === ADDITIONAL TERMS FROM TEXT ===
         "master archer": "arrow master",
         "keen eyesight": "hawk-like vision",
         "sensitive hearing": "acute hearing",
         "excellent bowmanship": "peerless archery",
         "long-held differences": "ancient rivalry",
         "messenger": "emissary",
         "Elvenking": "Star-lord",
         "prince": "scion",
         "woodland": "forest realm",
         "nimbly": "with grace",
         "snow": "white blanket",
         "Wargs": "shadow wolves",
         "arrows": "shafts",
         "Galadhrim": "Golden Grove guardians",
         "fellbeast": "winged terror",
         "masterful shot": "true aim",
         "song of lament": "dirge",
         "grey horse": "ash-steed",
         "Arod": "Swiftwind",
         "the White": "the Illuminated",

        # === Correction ===
        "The Lord of the Rings": "The Story"
        }


    def clean_text(self, text):
        """Очистка текста от HTML и лишнего форматирования"""
        # Удаление HTML тегов
        clean = re.compile('<.*?>')
        text = re.sub(clean, '', text)

        # Удаление квадратных скобок и их содержимого (ссылки в вики)
        text = re.sub(r'\[.*?\]', '', text)

        # Удаление фигурных скобок и их содержимого (шаблоны)
        text = re.sub(r'\{.*?\}', '', text)

        # Удаление лишних пробелов и переносов строк
        text = re.sub(r'\n+', '\n', text)
        text = re.sub(r' +', ' ', text)

        # Удаление специальных символов вики
        text = text.replace("'''", "").replace("''", "")

        return text.strip()

    def scrape_page(self, page_name):
        """Скачивание и очистка страницы"""
        try:
            url = self.base_url + quote(page_name)
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }

            response = requests.get(url, headers=headers)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')

            # Находим основной контент статьи
            content_div = soup.find('div', {'class': 'mw-parser-output'})
            if not content_div:
                return None

            # Удаляем ненужные элементы
            for element in content_div.find_all(['script', 'style', 'table', 'div.portable-infobox']):
                element.decompose()

            # Извлекаем заголовок
            title = soup.find('h1', {'class': 'page-header__title'})
            title_text = title.text.strip() if title else page_name.replace('_', ' ')

            # Извлекаем параграфы
            paragraphs = []
            for p in content_div.find_all('p'):
                text = self.clean_text(p.get_text())
                if text and len(text) > 50:  # Фильтруем слишком короткие параграфы
                    paragraphs.append(text)

            return {
                'title': title_text,
                'content': '\n\n'.join(paragraphs[:10])  # Берем первые 10 параграфов
            }

        except Exception as e:
            print(f"Ошибка при скачивании {page_name}: {e}")
            return None

    def replace_terms(self, text):
        """Замена терминов согласно словарю"""
        replaced_text = text

        # Сортируем термины по длине (от самых длинных к коротким) чтобы избежать частичных замен
        sorted_terms = sorted(self.terms_map.keys(), key=len, reverse=True)

        for term in sorted_terms:
            replacement = self.terms_map[term]
            # Используем регулярное выражение для замены с учетом регистра
            pattern = re.compile(re.escape(term), re.IGNORECASE)
            replaced_text = pattern.sub(replacement, replaced_text)

        return replaced_text

    def save_document(self, original_title, content, is_replaced=False):
        """Сохранение документа в файл"""
        # Создаем безопасное имя файла
        if is_replaced:
            filename = original_title.replace(' ', '_').replace('/', '_')
            filepath = os.path.join(self.knowledge_base_dir, f"{filename}.txt")
        else:
            filename = original_title.replace(' ', '_').replace('/', '_')
            filepath = os.path.join(self.knowledge_base_dir, f"{filename}.txt")

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"Title: {original_title}\n\n")
            f.write(content)

        return filepath

    def process_all_pages(self):
        """Основной метод обработки всех страниц"""
        print("Создание директории для базы знаний...")
        os.makedirs(self.knowledge_base_dir, exist_ok=True)

        print("Создание словаря замен...")
        self.create_custom_terms_map()

        successful_pages = 0

        for i, page_name in enumerate(self.pages_to_scrape, 1):
            print(f"Обработка страницы {i}/{len(self.pages_to_scrape)}: {page_name}")

            # Скачивание и очистка страницы
            page_data = self.scrape_page(page_name)

            if page_data and page_data['content']:

                # Замена терминов
                replaced_content = self.replace_terms(page_data['content'])
                replaced_title = self.replace_terms(page_data['title'])

                # Сохранение измененного текста
                replaced_path = self.save_document(
                    replaced_title,
                    replaced_content,
                    is_replaced=True
                )

                successful_pages += 1
                print(f"  ✓ Успешно обработано: {page_data['title']}")

            else:
                print(f"  ✗ Не удалось обработать: {page_name}")

            # Пауза между запросами
            time.sleep(1)

        # Сохранение словаря замен
        terms_map_path = os.path.join("", "terms_map.json")
        with open(terms_map_path, 'w', encoding='utf-8') as f:
            json.dump(self.terms_map, f, ensure_ascii=False, indent=2)

        print(f"\nОбработка завершена!")
        print(f"Успешно обработано страниц: {successful_pages}")

        return successful_pages

def main():
    """Основная функция"""
    processor = LotrFandomProcessor()

    try:
        successful_count = processor.process_all_pages()

        if successful_count > 0:
            print(f"\nСоздана уникальная база знаний с {successful_count} документами")
            print("Каждый документ сохранен в двух версиях:")
            print("  - *_original.txt - оригинальный очищенный текст")
            print("  - *_replaced.txt - текст с замененными терминами")
        else:
            print("Не удалось обработать ни одной страницы")

    except Exception as e:
        print(f"Произошла ошибка: {e}")

if __name__ == "__main__":
    main()
