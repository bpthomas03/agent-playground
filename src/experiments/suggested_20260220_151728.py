import re
from bs4 import BeautifulSoup
import csv

# Adjust the path as needed
INPUT_HTML = "data/search_dreams.html"
OUTPUT_CSV = "data/clean_dreams.csv"

def clean_highlights(text):
    # Remove <span class="highlight">...</span> tags and keep text inside
    return re.sub(r'<span class="highlight">(.*?)</span>', r'\1', text)

def clean_comments(text):
    # Remove <span class="comment">...</span> or broken ones (sometimes there is invalid HTML)
    text = re.sub(r'<span class="comment">.*?</span>', '', text, flags=re.DOTALL)
    text = re.sub(r'&lt;.*?&gt;', '', text)
    return text

def extract_word_count(soup):
    # Finds the (N words) span, returns int N if found
    word_span = soup.find("span", style=lambda x: x and "font-size:0.85em" in x)
    if word_span:
        m = re.search(r'\((\d+) words\)', word_span.text)
        if m:
            return int(m.group(1))
    return None

def parse_dream_blocks(html_text):
    # Each dream starts with <input type="checkbox" name="d" value="...">
    # We'll chunk the file based on this.
    pattern = r'(<input type="checkbox" name="d"[^>]+>.*?)(?=<input type="checkbox" name="d"|$)'
    blocks = re.findall(pattern, html_text, flags=re.DOTALL)
    return blocks

def parse_dream_block(block_html):
    soup = BeautifulSoup(block_html, "html.parser")
    checkbox = soup.find("input", {"name": "d"})
    label = soup.find("label")
    # Extract dream ID, subject, and date from label
    # example label: Barb Sanders: #0003 (1960-08-04)
    if not label:
        return None
    label_text = clean_highlights(str(label))
    label_text = BeautifulSoup(label_text, "html.parser").get_text()
    m = re.match(r'(.*?): #(\d+) \((\d{4}-\d{2}-\d{2})\)', label_text)
    if m:
        subject, dream_num, date = m.group(1), m.group(2), m.group(3)
    else:
        # fallback: try to extract at least subject and date
        subject = label_text.strip()
        dream_num = ""
        date = ""
        m = re.search(r'\((\d{4}-\d{2}-\d{2})\)', label_text)
        if m:
            date = m.group(1)
    # After the double <br> tag is the main narrative; get next element after label
    # or get text after <br><br ...>
    match_br = re.search(r'</label><br><br[^>]*>([\s\S]*?)(<span style="font-size:0.85em|$)', block_html)
    if match_br:
        narrative_html = match_br.group(1)
    else:
        # fallback to looking for the <span ...> line
        narrative_html = ""
    narrative_html = clean_highlights(narrative_html)
    narrative_html = clean_comments(narrative_html)
    narrative_text = BeautifulSoup(narrative_html, "html.parser").get_text().strip()
    # Handle any odd substitutions if necessary
    narrative_text = re.sub(r'\s+', ' ', narrative_text)
    # Extract word count
    count = extract_word_count(soup)
    # Extract dream ID from checkbox value if available
    dream_id = checkbox['value'] if checkbox and 'value' in checkbox.attrs else ""
    return {
        "dream_id": dream_id,
        "subject": subject,
        "date": date,
        "narrative": narrative_text,
        "word_count": count
    }

def open_html_file(filename):
    # Try utf-8, then latin-1 fallback for possible encoding issues.
    try:
        with open(filename, encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        with open(filename, encoding="latin-1") as f:
            return f.read()

html_text = open_html_file(INPUT_HTML)

dream_blocks = parse_dream_blocks(html_text)
records = []
for block in dream_blocks:
    rec = parse_dream_block(block)
    if rec:
        records.append(rec)

# Save to CSV
with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=["dream_id", "subject", "date", "narrative", "word_count"])
    writer.writeheader()
    for rec in records:
        writer.writerow(rec)

print(f"Saved {len(records)} dreams to {OUTPUT_CSV}")