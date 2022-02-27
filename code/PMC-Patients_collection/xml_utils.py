import xml.etree.cElementTree as ET
import json
import re
from tqdm import trange, tqdm
import os

# Element node types to be extracted within a paragragh, section or inline.
p_nodes = ['p', 'list-item', 'disp-quote', "AbstractText"]
sec_nodes = ['sec', 'list']
inline_elements = ['italic', 'bold', 'sup', 'strike', 'sub', 'sc', 'named-content', 'underline', \
    'statement', 'monospace', 'roman', 'overline', 'styled-content']

# Detect label in titles, such as "3.1" in "3.1 Patient 1"
label_pattern = re.compile(r'^[0-9]\.?[0-9]?\.?[0-9]?\.? ?')
# Replace several kinds of whitespace character into ' ' for convieniece of further processing. 
space = r"[\u3000\u2009\u2002\u2003\u00a0\u200a\xa0]"

"""
    Deal with unexpected and rebundant whitespace.
    Input:
        text: text to be cleaned.
    Output:
        cleaned text.
"""
def clean_text(text):
    text = re.sub(space, ' ', text).replace(u'\u2010', '-').strip()
    text = re.sub(r" +", ' ', text)
    text = re.sub(r"\n+", "\n", text)
    return text

"""
    Extract title in a title node.
    Input:
        sec: An element node of type 'sec'.
    Output:
        title of the section or string "" if no title node detected.
"""
def getTitle(sec):
    for child in sec:
        if child.tag == "title":
            title = getText(child)
            return clean_text(re.sub(label_pattern, '', title))
    return ""

"""
    Extract text from a given node and its successive children.
    Input:
        para: An element node of type 'p' or others.
    Output:
        text within the node and its successive children.
""" 
def getText(para):
    text = para.text if para.text else ""
    for child in para:
        if child.tag in inline_elements:
            text += child.text if child.text else ""
        if child.tag in sec_nodes or child.tag in p_nodes:
            text += getText(child) + ' '
        text += child.tail if child.tail else ""
    
    return clean_text(text)

"""
    Parse paragraph for an article or section. 
    Input:
        body: Element node to be parsed. If an article is to be parsed, input body node of xml.
        secname: Section name that will be concatenated ahead to all parsed paragraph.
            If an article is to be parsed, input empty string.
    Output:
        List of tuples (titles, text), where titles are section names (if subsection involved, titles are seperated by '[SEP]' token).
"""
def parse_paragraph(body, secname = ""):
    results = []
    title = getTitle(body)
    titles = secname + title
    if title:
        titles += "[SEP]"
    for child in body:
        if child.tag in p_nodes:
            text = getText(child)
            if len(text) > 1:
                results.append((titles, text))
        if child.tag in sec_nodes:
            results += parse_paragraph(child, titles)

    return results

"""
    Extract text of a section.
    Input:
        sec: Element node of type 'sec'.
    Output:
        Texts within this section, paragraphs seperated by '\n'.
"""
def getSection(sec):
    paras = parse_paragraph(sec)
    text = ""
    for para in paras:
        text += para[1] + '\n'
    return clean_text(text)


'''
if __name__ == "__main__":    
    cases = json.load(open("../../meta_data/PMC-Patients.txt", "r"))
    directory = "../../../PMC_OA"

    for case in tqdm(cases):
        file_name = case['file_name']
        tree = ET.parse(os.path.join(directory, file_name))
        root = tree.getroot()
        body = root.find(".//body")
        paras = parse_paragraph('', body)
        print(file_name)
        for para in paras:
            print(para[0])
            print(para[1])
        import ipdb; ipdb.set_trace()
'''