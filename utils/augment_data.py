import random
import re

# typos to augment the input data
KEYBOARD_MAP = {
    'a': 'qwsz', 'b': 'vghn', 'c': 'xdfv', 'd': 'ersfcx', 'e': 'wsdr',
    'f': 'rtgdvc', 'g': 'tyfhbv', 'h': 'yugjbn', 'i': 'ujko', 'j': 'uikhmn',
    'k': 'ijolm', 'l': 'kop', 'm': 'njk', 'n': 'bhjm', 'o': 'iklp', 'p': 'ol',
    'q': 'wa', 'r': 'edft', 's': 'awedxz', 't': 'rfgy', 'u': 'yhji',
    'v': 'cfgb', 'w': 'qase', 'x': 'zsdc', 'y': 'tghu', 'z': 'asx'
}

# typo injection for natural language
def inject_typos(text, prob=0.05):
    chars = list(text)
    for i in range(len(chars)):
        if random.random() < prob:
            choice = random.choice(["swap", "delete", "insert", "replace"])
            if choice == "swap" and i < len(chars) - 1:
                chars[i], chars[i+1] = chars[i+1], chars[i]
            elif choice == "delete":
                chars[i] = ""
            elif choice == "insert":
                chars[i] += random.choice("abcdefghijklmnopqrstuvwxyz")
            elif choice == "replace" and chars[i].lower() in KEYBOARD_MAP:
                chars[i] = random.choice(KEYBOARD_MAP[chars[i].lower()])
    return "".join(chars)


# typo injection for python code
def corrupt_python_code(code, prob=0.05):
    def typo_comment_or_string(match):
        return inject_typos(match.group(0), prob)

    # Corrupt only comments and string literals
    code = re.sub(r"#.*", typo_comment_or_string, code)
    code = re.sub(r"(['\"])(?:(?=(\\?))\2.)*?\1", typo_comment_or_string, code)
    return code

def corrupt_markdown(markdown, prob=0.05):
    def typo_comment_or_string(match):
        return inject_typos(match.group(0), prob)

    # Corrupt only comments and string literals
    markdown = re.sub(r"#.*", typo_comment_or_string, markdown)
    markdown = re.sub(r"(['\"])(?:(?=(\\?))\2.)*?\1", typo_comment_or_string, markdown)
    return markdown

# process 
def process_python(example):
    clean = example.get("content", "")
    noisy = corrupt_python_code(clean)
    return {"input": noisy, "target": clean}

def process_markdown(example):
    clean = example.get("content", "")
    noisy = corrupt_markdown(clean)
    return {"input": noisy, "target": clean}

def process_text(example):
    clean = example.get("content", "") if "content" in example else example.get("text", "")
    noisy = inject_typos(clean)
    return {"input": noisy, "target": clean}