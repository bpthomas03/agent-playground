from bs4 import BeautifulSoup
import re

# Load and parse HTML
with open("data/search_dreams.html", encoding="windows-1252") as f:
    soup = BeautifulSoup(f, "html.parser")

dreams = []

# Find all dream entries by locating input checkboxes with name="d"
for checkbox in soup.find_all("input", {"type": "checkbox", "name": "d"}):
    # Metadata from input
    dream_id = checkbox["value"]
    label = checkbox.find_next("label")
    label_text = label.get_text(separator="", strip=True)
    
    # Extract dreamer/source, number, and date from label
    meta_match = re.match(r"(.+?):\s*#(\d+)\s*\((\d{4}-\d{2}-\d{2})\)", label_text)
    if meta_match:
        source = meta_match.group(1)
        number = meta_match.group(2)
        date = meta_match.group(3)
    else:
        source = label_text
        number = ""
        date = ""
    
    # Dream text is after two <br> tags (br + br), styled with margin hack, then the text up to <span> (word count/comment), then <hr>
    # Find the <br> (with margin-bottom) after label
    br = label.find_next("br", attrs={"style": True})
    if not br:
        continue
    # Dream text is the next_sibling after that <br>
    raw_text = ""
    next_node = br.next_sibling
    while next_node:
        # If we hit a <span> (word count), stop
        if getattr(next_node, "name", None) == "span":
            break
        # If node is a tag, get its text, else use as is
        if hasattr(next_node, "get_text"):
            raw_text += next_node.get_text(" ", strip=True)
        elif isinstance(next_node, str):
            raw_text += next_node.strip()
        next_node = next_node.next_sibling

    # Remove highlighted <span class="highlight"> tags: get full text, don't keep markup
    text_soup = BeautifulSoup(raw_text, "html.parser")
    for h in text_soup.find_all("span", class_="highlight"):
        h.unwrap()
    for c in text_soup.find_all("span", class_="comment"):
        c.unwrap()
    clean_text = text_soup.get_text(" ", strip=True)

    dreams.append({
        "id": dream_id,
        "source": source,
        "number": number,
        "date": date,
        "text": clean_text
    })

# Example: Print as JSON lines for further analysis
import json
for dream in dreams:
    print(json.dumps(dream, ensure_ascii=False))